/*
Distance driven 3D, branchless version.
It requires double precision of the GPU.
*/

#include "distanceDriven.h"
#include "distanceDriven3DUtils.h"
#include "cudaMath.h"
#include "siddon.h"

#include <stdexcept>
#include <exception>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;
/* 
accumulate the pixel value along x axis
use double precision for dst because potentially a lot of pixels will be added
dst[0,y,z] = 0, dst[1,y,z] = src[0,y,z], dst[2,y,z] = src[0,y,z]+src[1,y,z], ...

dst - the buffer to receive the accumulation, of size (nx+1, ny+1, nz)
src - the original image, of size (nx, ny, nz)
nx,ny,nz - the dimension of the src image
*/
__global__ void AccumulateXYAlongXKernel(double* dst, const float* src, size_t nx, size_t ny, size_t nz)
{
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int iz = blockDim.z * blockIdx.z + threadIdx.z;

	if (iy >= ny || iz >= nz)
	{
		return;
	}

	dst += iz * (nx + 1) * (ny + 1) + (iy + 1) * (nx + 1);	// skip iy == 0, which should always be zero in dst
	src += iz * nx * ny + iy * nx;
	dst[0] = 0;
	for (int ix = 0; ix < nx; ix++)
	{
		dst[ix + 1] = dst[ix] + src[ix];
	}
}

/*
accumulate the pixel value along y axis
use double precision for dst because potentially a lot of pixels will be added
the function should be called after AccumulateXYAlongXKernel. it directly accumulate inplace within the buffer

acc - the buffer for the accumulation, of size (nx+1, ny+1, nz)
nx,ny,nz - the dimension of the src image
*/
__global__ void AccumulateXYAlongYKernel(double* acc, size_t nx, size_t ny, size_t nz)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iz = blockDim.z * blockIdx.z + threadIdx.z;

	if (ix >= nx || iz >= nz)
	{
		return;
	}

	acc += iz * (nx + 1) * (ny + 1) + ix + 1;	// skip ix == 0, which should always be zero in dst
	
	for (int iy = 0; iy < ny; iy++)
	{
		acc[(iy + 1) * (nx + 1)] = acc[(iy + 1) * (nx + 1)] + acc[iy * (nx + 1)];
	}
}

/*
Distance driven projection. The projection should always be performed with z axis as the main axis.

iviews - the list of iview where DDFP is performed on the XY plane
nValidViews - length of iviews
*/
__global__ void DDFPConeKernelXY(
	float* pPrjs,
	const double* acc,
	const int* iviews,
	int nValidViews,
	int nview,
	const float3* pDetCenter,
	const float3* pDetU,
	const float3* pDetV,
	const float3* pSrc,
	const Grid grid,
	const Detector det
)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iv = blockDim.y * blockIdx.y + threadIdx.y;
	int ind = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || ind >= nValidViews || iv >= det.nv)
	{
		return;
	}

	int iview = iviews[ind];

	// coordinates of the center point of each edge of the detector's unit
	// dstx1, dstx2 are the edges at +- u along x axis
	// dsty1, dsty2 are the edges at +- v along y axis
	float u = (iu - det.off_u - (det.nu - 1) / 2.0f) * det.du;
	float v = (iv - det.off_v - (det.nv - 1) / 2.0f) * det.dv;
	float3 dstx1 = UVToCart(u - det.du / 2, v, pDetCenter[iview], pDetU[iview], pDetV[iview]);
	float3 dstx2 = UVToCart(u + det.du / 2, v, pDetCenter[iview], pDetU[iview], pDetV[iview]);
	float3 dsty1 = UVToCart(u, v - det.dv / 2, pDetCenter[iview], pDetU[iview], pDetV[iview]);
	float3 dsty2 = UVToCart(u, v + det.dv / 2, pDetCenter[iview], pDetU[iview], pDetV[iview]);

	float3 src = pSrc[iview];

	// convert to image coordinate
	src = PhysicsToImg(src, grid);
	dstx1 = PhysicsToImg(dstx1, grid);
	dstx2 = PhysicsToImg(dstx2, grid);
	dsty1 = PhysicsToImg(dsty1, grid);
	dsty2 = PhysicsToImg(dsty2, grid);

	// make sure dstx1.x < dstx2.x
	if (dstx1.x > dstx2.x)
	{
		float3 tmp = dstx1;
		dstx1 = dstx2;
		dstx2 = tmp;
	}

	// make sure dsty1.y < dsty2.y
	if (dsty1.y > dsty2.y)
	{
		float3 tmp = dsty1;
		dsty1 = dsty2;
		dsty2 = tmp;
	}

	float val = 0;
	float rx1 = (dstx1.x - src.x) / (dstx1.z - src.z);
	float rx2 = (dstx2.x - src.x) / (dstx2.z - src.z);
	float ry1 = (dsty1.y - src.y) / (dsty1.z - src.z);
	float ry2 = (dsty2.y - src.y) / (dsty2.z - src.z);

	// calculate intersection with each xy plane at different z
	for (int iz = 0; iz < grid.nz; iz++)
	{
		float x1 = src.x + rx1 * (iz - src.z);
		float x2 = src.x + rx2 * (iz - src.z);
		float y1 = src.y + ry1 * (iz - src.z);
		float y2 = src.y + ry2 * (iz - src.z);

		val += (
			InterpolateXY(acc, x2, y2, iz, grid.nx + 1, grid.ny + 1, grid.nz)
			+ InterpolateXY(acc, x1, y1, iz, grid.nx + 1, grid.ny + 1, grid.nz)
			- InterpolateXY(acc, x2, y1, iz, grid.nx + 1, grid.ny + 1, grid.nz)
			- InterpolateXY(acc, x1, y2, iz, grid.nx + 1, grid.ny + 1, grid.nz)
		) / ((x2 - x1) * (y2 - y1));

	}

	// normalize by length
	// use physics coordinate
	float3 dst = UVToCart(u, v, pDetCenter[iview], pDetU[iview], pDetV[iview]);
	src = pSrc[iview];
	val *= grid.dz / fabsf((src.z - dst.z)) * sqrtf(
		(src.z - dst.z) * (src.z - dst.z) + (src.y - dst.y) * (src.y-dst.y) + (src.x - dst.x) * (src.x - dst.x)
	);

	pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] = val;

}

// no textures, use double-precision software interpolation
void DistanceDrivenTomo::ProjectionTomoBranchless(
	const float* pcuImg,
	float* pcuPrj,
	const float* pcuDetCenter,
	const float* pcuSrc,
	double* pcuAcc,
	const int* pcuIviews,
	const float3* pcuDetU,
	const float3* pcuDetV
)
{
	try
	{
		Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
		Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

		dim3 threadX(1,32,16);
		dim3 blockX(1, ceilf(ny / 32.f), ceilf(nz / 16.f));
		dim3 threadY(32,1,16);
		dim3 blockY(ceilf(nx / 32.f), 1, ceilf(nz / 16.f));
		dim3 threads(32, 16, 1);
		dim3 blocks(ceilf(nu / 32.f), ceilf(nv / 16.f), nview);
		for (int ib = 0; ib < nBatches; ib++)
		{
			// step 1: calculate accumulated images
			// pAcc has the dimension in order (z, y, x)
			cudaMemset(pcuAcc, 0, sizeof(double) * (nx + 1) * (ny + 1) * nz);
			AccumulateXYAlongXKernel<<<blockX, threadX>>>(pcuAcc, pcuImg + ib * nx * ny * nz , nx, ny, nz);
			cudaDeviceSynchronize();
			AccumulateXYAlongYKernel<<<blockY, threadY>>>(pcuAcc, nx, ny, nz);
			cudaDeviceSynchronize();

			// step 2: interpolation
			DDFPConeKernelXY<<<blocks, threads>>>(
				pcuPrj + ib * nu * nv * nview,
				pcuAcc + ib * (nx + 1) * (ny + 1) * nz,
				pcuIviews,
				nview,
				nview,
				(const float3*)pcuDetCenter,
				pcuDetU,
				pcuDetV,
				(const float3*)pcuSrc,
				grid,
				det
			);

			cudaDeviceSynchronize();
		}

	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "DistanceDrivenTomo::ProjectionTomoBranchless Error: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}
}

// dst has the dimension (batch, nview, nv+1, nu+1)
// src has the dimension (batch, nview, nv, nu)
__global__ void AccumulateUVAlongUKernel(double* dst, const float* src, size_t nu, size_t nv, size_t nview)
{
	int iv = blockDim.y * blockIdx.y + threadIdx.y;
	int iview = blockDim.z * blockIdx.z + threadIdx.z;

	if (iview >= nview || iv >= nv)
	{
		return;
	}

	dst += iview * (nu + 1) * (nv + 1) + (iv + 1) * (nu + 1);	// skip iv == 0, which should always be zero in dst
	src += iview * nu * nv + iv * nu;
	dst[0] = 0;
	for (int iu = 0; iu < nu; iu++)
	{
		dst[iu + 1] = dst[iu] + src[iu];
	}
}

// this kernel should be called right after AccumulateUVAlongUKernel to integrate along y axis,
// acc has the dimension (batch, nview, nv+1, nu+1)
__global__ void AccumulateUVAlongVKernel(double* acc, size_t nu, size_t nv, size_t nview)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iview = blockDim.z * blockIdx.z + threadIdx.z;

	if (iu >= nu || iview >= nview)
	{
		return;
	}

	acc += iview * (nu + 1) * (nv + 1) + iu + 1;	// skip iu == 0, which should always be zero in dst
	for (int iv = 0; iv < nv; iv++)
	{
		acc[(iv + 1) * (nu + 1)] = acc[(iv + 1) * (nu + 1)] + acc[iv * (nu + 1)];
	}
}

// BP when detector aligns with the cartesian coordinate
__global__ void DDBPConeCartKernelXY(
	float* pImg,
	const double* acc,
	const int* iviews,
	size_t nValidViews,
	size_t nview,
	const float3* pDetCenter,
	const float3* pSrc,
	const Grid grid,
	const Detector det
)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= grid.nx || iy >= grid.ny || iz >= grid.nz)
	{
		return;
	}

	float x = (ix - (grid.nx - 1) / 2.0f) * grid.dx + grid.cx;
	float y = (iy - (grid.ny - 1) / 2.0f) * grid.dy + grid.cy;
	float z = (iz - (grid.nz - 1) / 2.0f) * grid.dz + grid.cz;

	float val = 0;
	for (int ind = 0; ind < nValidViews; ind++)
	{
		int iview = iviews[ind];
		float3 src = pSrc[iview];
		float3 detCenter = pDetCenter[iview];

		float u1 = ProjectConeToDetCart(make_float3(x - grid.dx / 2, y, z), detCenter, src, det).x;
		float u2 = ProjectConeToDetCart(make_float3(x + grid.dx / 2, y, z), detCenter, src, det).x;
		float v1 = ProjectConeToDetCart(make_float3(x, y - grid.dy / 2, z), detCenter, src, det).y;
		float v2 = ProjectConeToDetCart(make_float3(x, y + grid.dy / 2, z), detCenter, src, det).y;


		val += (
			InterpolateXY(acc, u2, v2, iview, det.nu + 1, det.nv + 1, nview)
			- InterpolateXY(acc, u2, v1, iview, det.nu + 1, det.nv + 1, nview)
			+ InterpolateXY(acc, u1, v1, iview, det.nu + 1, det.nv + 1, nview)
			- InterpolateXY(acc, u1, v2, iview, det.nu + 1, det.nv + 1, nview)
		) / ((u2 - u1) * (v2 - v1));

		// pImg[iz * grid.nx * grid.ny + iy * grid.nx + ix] = z - src.z;
		
	}

	pImg[iz * grid.nx * grid.ny + iy * grid.nx + ix] = val;

}

// no textures, use double-precision software interpolation
void DistanceDrivenTomo::BackprojectionTomoBranchless(
	float* pcuImg,
	const float* pcuPrj,
	const float* pcuDetCenter,
	const float* pcuSrc,
	float* pcuWeightedPrjs,
	double* pcuAcc,
	const int* pcuIviews
)
{
	try
	{
		Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
		Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

		// step 0: preweight the projections for ray intersection length
		dim3 threadUV(32, 16, 1);
		dim3 blockUV(ceilf(nu / 32.f), ceilf(nv / 16.f), 1);
		cudaMemcpy(pcuWeightedPrjs, pcuPrj, sizeof(float) * nBatches * nu * nv * nview, cudaMemcpyDeviceToDevice);
		for (int ib = 0; ib < nBatches; ib++)
		{
			PreweightBPCartKernelXY<<<blockUV, threadUV>>>(
				pcuWeightedPrjs + ib * nu * nv * nview,
				pcuIviews,
				nview,
				nview,
				(const float3*)pcuDetCenter,
				(const float3*)pcuSrc,
				grid.dz,
				det
			);
		}
		cudaDeviceSynchronize();

		dim3 threadU(1, 32, 4);
		dim3 blockU(1, ceilf(nv / 32.f), ceilf(nview / 4.f));
		dim3 threadV(32, 1, 4);
		dim3 blockV(ceilf(nu / 32.f), 1, ceilf(nview / 4.f));
		dim3 threadImg(32, 16, 1);
		dim3 blockImg(ceilf(nx / 32.f), ceilf(ny / 16.f), nz);
		for (int ib = 0; ib < nBatches; ib++)
		{
			cudaMemset(pcuAcc, 0, sizeof(double) * (nu + 1) * (nv + 1) * nview);

			// step 1: calculate accumulated projections
			AccumulateUVAlongUKernel<<<blockU, threadU>>>(pcuAcc, pcuWeightedPrjs + ib * nu * nv * nview, nu, nv, nview);
			cudaDeviceSynchronize();
			AccumulateUVAlongVKernel<<<blockV, threadV>>>(pcuAcc, nu, nv, nview);
			cudaDeviceSynchronize();

			// step 2: interpolation
			DDBPConeCartKernelXY<<<blockImg, threadImg>>>(
				pcuImg + ib * nx * ny * nz,
				pcuAcc,
				pcuIviews,
				nview,
				nview,
				(const float3*)pcuDetCenter,
				(const float3*)pcuSrc,
				grid,
				det
			);
			cudaDeviceSynchronize();
		}

	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "DistanceDrivenTomo::BackprojectionTomoBranchless Error: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}
}
