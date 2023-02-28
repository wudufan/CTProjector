/*
Distance driven 3D, box integral version.
It only requires single precision of the GPU.
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
Distance driven projection. The projection should always be performed with z axis as the main axis.
This kernel uses box integral thus no accumulation buffer is needed.

iviews - the list of iview where DDFP is performed on the XY plane
nValidViews - length of iviews
*/
__global__ void DDFPConeBoxIntKernelXY(
	float* pPrjs,
	const float* pImg,
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

	// Calculate intersection with each xy plane at different z
	// The value to be integrated should be the averaged value inside the intersecting box.
	for (int iz = 0; iz < grid.nz; iz++)
	{
		float x1 = src.x + rx1 * (iz - src.z);
		float x2 = src.x + rx2 * (iz - src.z);
		float y1 = src.y + ry1 * (iz - src.z);
		float y2 = src.y + ry2 * (iz - src.z);

		val += IntegralBoxXY(pImg, x1, y1, x2, y2, iz, grid.nx, grid.ny, grid.nz) / ((x2 - x1) * (y2 - y1));

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

void DistanceDrivenTomo::ProjectionTomoBoxInt(
	const float* pcuImg,
	float* pcuPrj,
	const float* pcuDetCenter,
	const float* pcuSrc,
	const int* pcuIviews,
	const float3* pcuDetU,
	const float3* pcuDetV
)
{
	try
	{
		Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
		Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

		dim3 threads(32, 16, 1);
		dim3 blocks(ceilf(nu / 32.f), ceilf(nv / 16.f), nview);
		for (int ib = 0; ib < nBatches; ib++)
		{
			// Box integral
			DDFPConeBoxIntKernelXY<<<blocks, threads>>>(
				pcuPrj + ib * nu * nv * nview,
				pcuImg + ib * nx * ny * nz,
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
		oss << "DistanceDrivenTomo::ProjectionTomoBoxInt Error: " << e.what() 
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}

}


// BP when detector aligns with the cartesian coordinate
__global__ void DDBPConeBoxIntCartKernelXY(
	float* pImg,
	const float* pPrj,
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


		val += IntegralBoxXY(pPrj, u1, v1, u2, v2, iview, det.nu, det.nv, nview) / ((u2 - u1) * (v2 - v1));

		// pImg[iz * grid.nx * grid.ny + iy * grid.nx + ix] = z - src.z;
		
	}

	pImg[iz * grid.nx * grid.ny + iy * grid.nx + ix] = val;

}

// no textures, use double-precision software interpolation
void DistanceDrivenTomo::BackprojectionTomoBoxInt(
	float* pcuImg,
	const float* pcuPrj,
	const float* pcuDetCenter,
	const float* pcuSrc,
	float* pcuWeightedPrjs,
	const int* pcuIviews
)
{
	try
	{
		Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
		Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

		// step 0: preweight the projections for ray intersection length
		dim3 threadUV(32, 16, 1);
		dim3 blockUV(ceilf(nu / 32.f), ceilf(nv / 16.f), nview);
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

		dim3 threadImg(32, 16, 1);
		dim3 blockImg(ceilf(nx / 32.f), ceilf(ny / 16.f), nz);
		for (int ib = 0; ib < nBatches; ib++)
		{
			// box integral
			DDBPConeBoxIntCartKernelXY<<<blockImg, threadImg>>>(
				pcuImg + ib * nx * ny * nz,
				pcuWeightedPrjs,
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
		oss << "DistanceDrivenTomo::BackprojectionTomoBoxInt Error: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}
}


/*
Distance driven projection. The projection should always be performed with x or y as the main axis.
This kernel uses box integral thus no accumulation buffer is needed.
*/
/*
__global__ void DDFPConeBoxIntKernelHelical(
	float* pPrjs,
	const float* pImg,
	const float* pDeg,
	const float* pZ,
	size_t nview,
	const Grid grid,
	const Detector det,
	float dsd,
	float dso
)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iv = blockDim.y * blockIdx.y + threadIdx.y;
	int iview = blockIdx.z * blockDim.z + threadIdx.z;

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

	// Calculate intersection with each xy plane at different z
	// The value to be integrated should be the averaged value inside the intersecting box.
	for (int iz = 0; iz < grid.nz; iz++)
	{
		float x1 = src.x + rx1 * (iz - src.z);
		float x2 = src.x + rx2 * (iz - src.z);
		float y1 = src.y + ry1 * (iz - src.z);
		float y2 = src.y + ry2 * (iz - src.z);

		val += IntegralBoxXY(pImg, x1, y1, x2, y2, iz, grid.nx, grid.ny, grid.nz) / ((x2 - x1) * (y2 - y1));

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
*/