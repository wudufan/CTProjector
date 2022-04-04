#include "distanceDriven.h"
#include "cudaMath.h"
#include "projector.h"

#include <stdexcept>
#include <exception>
#include <sstream>
#include <iostream>

using namespace std;

void DistanceDrivenFan::Allocate(bool forward, bool backward)
{
	this->Free();

	if (forward)
	{
		if (cudaSuccess != cudaMalloc(&pAccX, sizeof(float) * (nx + 1) * ny * nz))
		{
			throw runtime_error("pAccX allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pAccY, sizeof(float) * nx * (ny + 1) * nz))
		{
			throw runtime_error("pAccY allocation failed");
		}
	}
	if (backward)
	{
		if (cudaSuccess != cudaMalloc(&pWeightedPrjs, sizeof(float) * nu * nv * nview))
		{
			throw runtime_error("pWeightedPrjs allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pAccU, sizeof(float) * (nu + 1) * nv * nview))
		{
			throw runtime_error("pAccU allocation failed");
		}
	}
	
}

void DistanceDrivenFan::Free()
{
	if (pAccX != NULL) cudaFree(pAccX);
	if (pAccY != NULL) cudaFree(pAccY);
	if (pAccU != NULL) cudaFree(pAccU);
	if (pWeightedPrjs != NULL) cudaFree(pWeightedPrjs);

	pAccX = NULL;
	pAccY = NULL;
	pAccU = NULL;
	pWeightedPrjs = NULL;
}

__global__ void AccumulateKernelX(float* dst, const float* src, size_t nx, size_t ny, size_t nz)
{
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int iz = blockDim.z * blockIdx.z + threadIdx.z;

	if (iy >= ny || iz >= nz)
	{
		return;
	}

	dst += iz * (nx + 1) * ny + iy * (nx + 1);
	src += iz * nx * ny + iy * nx;
	dst[0] = 0;
	for (int ix = 0; ix < nx; ix++)
	{
		dst[ix + 1] = dst[ix] + src[ix];
	}
}

__global__ void AccumulateKernelY(float* dst, const float* src, size_t nx, size_t ny, size_t nz)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iz = blockDim.z * blockIdx.z + threadIdx.z;

	if (ix >= nx || iz >= nz)
	{
		return;
	}

	dst += iz * nx * (ny + 1) + ix;
	src += iz * nx * ny + ix;
	dst[0] = 0;
	for (int iy = 0; iy < ny; iy++)
	{
		dst[(iy + 1) * nx] = dst[iy * nx] + src[iy * nx];
	}
}

// Trace the distance driven forward projection
__device__ float DDFPTracing(
	const float* pAccX,
	const float* pAccY,
	float2 src1,
	float2 src2,
	float2 dst1,
	float2 dst2,
	float z,
	float deg,
	float a,
	float cosDeg,
	float sinDeg,
	const Grid& grid
)
{
	float val = 0;
	// calculate the intersection with at each y
	if (fabsf(cosDeg) > fabsf(sinDeg))
	{
		// calculate the intersection with at each y
		float r1 = (dst1.x - src1.x) / (dst1.y - src1.y);
		float r2 = (dst2.x - src2.x) / (dst2.y - src2.y);
		for (int iy = 0; iy < grid.ny; iy++)
		{
			float x1 = src1.x + r1 * (iy - src1.y);
			float x2 = src2.x + r2 * (iy - src2.y);

			/*
			The image coordinates has the corner of the first pixel as the (0,0,0);
			The x1, x2 here are in the corner coordinate;
			The iy, z here are in the pixel-center coordinate (notice that z has an offset of 0.5)
			paccx(z,iy,0) means the integral at x=0 (left of first pixel) for (iy, z), which is always 0 
			paccx(z,iy,1) means the integral at x=1 (left of second pixel) for (iy, z)

			For integer ix, paccx[ix]=integral(0 to ix), and paccx[ix+1] = integral(0 to ix+1)
			Because the pixel value is constant inside between ix and ix+1, linear interpolation at x (ix<x<ix+1) gives the integral(0,x)

			Hence, paccx(x2) - paccx(x1) gives the integral(ix1, ix2)

			The final distance driven is the average, so it should be nomalized by length (x2-x1)
			*/
			val += (
				InterpolateXZ(pAccX, x2, iy, z, grid.nx + 1, grid.ny, grid.nz) 
				- InterpolateXZ(pAccX, x1, iy, z, grid.nx + 1, grid.ny, grid.nz)
			) / (x2 - x1);
				
		}

		// normalize by length
		val *= grid.dy / fabsf(__cosf(deg - a));
	}
	else
	{
		// calculate the intersection with at each x
		float r1 = (dst1.y - src1.y) / (dst1.x - src1.x);
		float r2 = (dst2.y - src2.y) / (dst2.x - src2.x);
		for (int ix = 0; ix < grid.nx; ix++)
		{
			float y1 = src1.y + r1 * (ix - src1.x);
			float y2 = src2.y + r2 * (ix - src2.x);

			// Please see DDFPFanKernelX for detailed explanation
			val += (
				InterpolateYZ(pAccY, ix, y2, z, grid.nx, grid.ny + 1, grid.nz)
				- InterpolateYZ(pAccY, ix, y1, z, grid.nx, grid.ny + 1, grid.nz)
			) / (y2 - y1);
		}

		// normalize by length
		val *= grid.dx / fabsf(__sinf(deg - a));
	}

	return val;
}

/*
Distance driven fanbeam projection

pPrjs - projection of size [nview, nv, nu]
pAccX - accumulation of images along x, size [nz, ny, nx+1]
pAccY - accumulation of images along y, size [nz, ny+1, nx]
pDeg - projection angles, size [nview]
nviews - total view number
grid - image grid
det - detector information
dsd - distance between source and detector
dso - distance between source and iso-center

*/
__global__ void DDFPFanKernel(
	float* pPrjs,
	const float* pAccX,
	const float* pAccY,
	const float* pDeg,
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

	if (iu >= det.nu || iv >= det.nv || iview >= nview)
	{
		return;
	}

	float deg = pDeg[iview];

	float cosDeg = __cosf(deg);
	float sinDeg = __sinf(deg);
	float a = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float z = (iv - (det.nv-1) / 2.0f + det.off_v) * det.dv;

	// calculate the coordinates of the detector cell's edges
	float2 src = make_float2(dso * sinDeg, -dso * cosDeg);
	float2 dst1 = make_float2(dsd * __sinf(a - det.du / 2), -dso + dsd * __cosf(a - det.du / 2));
	float2 dst2 = make_float2(dsd * __sinf(a + det.du / 2), -dso + dsd * __cosf(a + det.du / 2));
	dst1 = make_float2(dst1.x * cosDeg - dst1.y * sinDeg, dst1.x * sinDeg + dst1.y * cosDeg);
	dst2 = make_float2(dst2.x * cosDeg - dst2.y * sinDeg, dst2.x * sinDeg + dst2.y * cosDeg);

	// convert to image coordinate
	src = PhysicsToImg(src, grid);
	dst1 = PhysicsToImg(dst1, grid);
	dst2 = PhysicsToImg(dst2, grid);
	z = (z - grid.cz) / grid.dz + grid.nz / 2.f - 0.5f;

	// make sure dst1.x < dst2.x
	if (dst1.x > dst2.x)
	{
		float2 temp = dst1;
		dst1 = dst2;
		dst2 = temp;
	}
	
	float val = DDFPTracing(
		pAccX, pAccY, src, src, dst1, dst2, z, deg, a, cosDeg, sinDeg, grid
	);

	pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] = val;

}


/*
Distance driven parallel projection

pPrjs - projection of size [nview, nv, nu]
pAccX - accumulation of images along x, size [nz, ny, nx+1]
pAccY - accumulation of images along y, size [nz, ny+1, nx]
pDeg - projection angles, size [nview]
nviews - total view number
grid - image grid
det - detector information

*/
__global__ void DDFPParallelKernel(
	float* pPrjs,
	const float* pAccX,
	const float* pAccY,
	const float* pDeg,
	size_t nview,
	const Grid grid,
	const Detector det
)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iv = blockDim.y * blockIdx.y + threadIdx.y;
	int iview = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iv >= det.nv || iview >= nview)
	{
		return;
	}

	float deg = pDeg[iview];

	float cosDeg = __cosf(deg);
	float sinDeg = __sinf(deg);
	float u = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float z = (iv - (det.nv-1) / 2.0f + det.off_v) * det.dv;

	// a virtual dso to put the src and dst
	float dso = grid.nx * grid.dx + grid.ny * grid.dy;

	// calculate the coordinates of the detector cell's edges
	float2 src1 = make_float2(u - det.du / 2, -dso);
	float2 src2 = make_float2(u + det.du / 2, -dso);
	float2 dst1 = make_float2(u - det.du / 2, dso);
	float2 dst2 = make_float2(u + det.du / 2, dso);
	src1 = make_float2(src1.x * cosDeg - src1.y * sinDeg, src1.x * sinDeg + src1.y * cosDeg);
	src2 = make_float2(src2.x * cosDeg - src2.y * sinDeg, src2.x * sinDeg + src2.y * cosDeg);
	dst1 = make_float2(dst1.x * cosDeg - dst1.y * sinDeg, dst1.x * sinDeg + dst1.y * cosDeg);
	dst2 = make_float2(dst2.x * cosDeg - dst2.y * sinDeg, dst2.x * sinDeg + dst2.y * cosDeg);

	// convert to image coordinate
	src1 = PhysicsToImg(src1, grid);
	src2 = PhysicsToImg(src2, grid);
	dst1 = PhysicsToImg(dst1, grid);
	dst2 = PhysicsToImg(dst2, grid);
	z = (z - grid.cz) / grid.dz + grid.nz / 2.f - 0.5f;

	// make sure dst1.x < dst2.x
	if (dst1.x > dst2.x)
	{
		float2 temp = dst1;
		dst1 = dst2;
		dst2 = temp;

		temp = src1;
		src1 = src2;
		src2 = temp;
	}
	
	float val = DDFPTracing(
		pAccX, pAccY, src1, src2, dst1, dst2, z, deg, 0, cosDeg, sinDeg, grid
	);

	pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] = val;

}

void DistanceDrivenFan::Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg)
{
	try
	{
		Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
		Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

		// step 1: calculate accumulated images
		dim3 threadX(1,256,1);
		dim3 blockX(1, ceilf(ny / 256.0f), nz);
		dim3 threadY(256,1,1);
		dim3 blockY(ceilf(nx / 256.0f), 1, nz);
		for (int ib = 0; ib < nBatches; ib++)
		{
			// AccX and pAccY has the dimension in order (batch, z, y, x)
			AccumulateKernelX<<<blockX, threadX, 0, m_stream>>>(pAccX, pcuImg + ib * nx * ny * nz, nx, ny, nz);
			AccumulateKernelY<<<blockY, threadY, 0, m_stream>>>(pAccY, pcuImg + ib * nx * ny * nz, nx, ny, nz);
			cudaStreamSynchronize(m_stream);
			
			// step 2: interpolation
			dim3 threadDet, blockDet;
			GetThreadsForXZ(threadDet, blockDet, nu, nv, nview);
			DDFPFanKernel<<<blockDet, threadDet, 0, m_stream>>>(
				pcuPrj + ib * nu * nv * nview, pAccX, pAccY, pcuDeg, nview, grid, det, dsd, dso
			);
			cudaStreamSynchronize(m_stream);
		}
	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "DistanceDrivenFan::Projection Error: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}
}


void DistanceDrivenParallel::Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg)
{
	try
	{
		Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
		Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

		// step 1: calculate accumulated images
		dim3 threadX(1,256,1);
		dim3 blockX(1, ceilf(ny / 256.0f), nz);
		dim3 threadY(256,1,1);
		dim3 blockY(ceilf(nx / 256.0f), 1, nz);
		for (int ib = 0; ib < nBatches; ib++)
		{
			// AccX and pAccY has the dimension in order (batch, z, y, x)
			AccumulateKernelX<<<blockX, threadX, 0, m_stream>>>(pAccX, pcuImg + ib * nx * ny * nz, nx, ny, nz);
			AccumulateKernelY<<<blockY, threadY, 0, m_stream>>>(pAccY, pcuImg + ib * nx * ny * nz, nx, ny, nz);
			cudaStreamSynchronize(m_stream);
			
			// step 2: interpolation
			dim3 threadDet, blockDet;
			GetThreadsForXZ(threadDet, blockDet, nu, nv, nview);
			DDFPParallelKernel<<<blockDet, threadDet, 0, m_stream>>>(
				pcuPrj + ib * nu * nv * nview, pAccX, pAccY, pcuDeg, nview, grid, det
			);
			// DDFPFanKernel<<<blockDet, threadDet, 0, m_stream>>>(
			// 	pcuPrj + ib * nu * nv * nview, pAccX, pAccY, pcuDeg, nview, grid, det, dsd, dso
			// );
			cudaStreamSynchronize(m_stream);
		}
	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "DistanceDrivenParallel::Projection Error: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}
}

extern "C" int cupyDistanceDrivenFanProjection(
	float* prj,
	const float* img,
	const float* deg,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	float dx,
	float dy,
	float dz,
	float cx,
	float cy,
	float cz,
	size_t nu,
	size_t nv,
	size_t nview,
	float da,
	float dv,
	float off_a,
	float off_v,
	float dsd,
	float dso
)
{
	try
	{
		DistanceDrivenFan projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, da, dv, off_a, off_v, dsd, dso
		);
		
		projector.Allocate(true, false);
		projector.Projection(img, prj, deg);
		projector.Free();
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cupyDistanceDrivenFanProjection() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();
}

extern "C" int cupyDistanceDrivenParallelProjection(
	float* prj,
	const float* img,
	const float* deg,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	float dx,
	float dy,
	float dz,
	float cx,
	float cy,
	float cz,
	size_t nu,
	size_t nv,
	size_t nview,
	float da,
	float dv,
	float off_a,
	float off_v
)
{
	try
	{
		DistanceDrivenParallel projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, da, dv, off_a, off_v, 1000, 500
		);
		
		projector.Allocate(true, false);
		projector.Projection(img, prj, deg);
		projector.Free();
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cupyDistanceDrivenParallelProjection() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();
}

extern "C" int cDistanceDrivenFanProjection(
	float* prj,
	const float* img,
	const float* deg,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	float dx,
	float dy,
	float dz,
	float cx,
	float cy,
	float cz,
	size_t nu,
	size_t nv,
	size_t nview,
	float da,
	float dv,
	float off_a,
	float off_v,
	float dsd,
	float dso
)
{
	float* pcuImg = NULL;
	float* pcuPrj = NULL;
	float* pcuDeg = NULL;
	try
	{
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nBatches * nx * ny * nz))
		{
			throw runtime_error("pcuImg allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nu * nv * nview))
		{
			throw runtime_error("pcuPrj allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuDeg, sizeof(float) * nview))
		{
			throw runtime_error("pcuDeg allocation failed");
		}

		cudaMemcpy(pcuDeg, deg, sizeof(float) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuImg, img, sizeof(float) * nBatches * nx * ny * nz, cudaMemcpyHostToDevice);
		cudaMemset(pcuPrj, 0, sizeof(float) * nBatches * nu * nv * nview);

		DistanceDrivenFan projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, da, dv, off_a, off_v, dsd, dso
		);
		
		projector.Allocate(true, false);
		projector.Projection(pcuImg, pcuPrj, pcuDeg);
		projector.Free();

		cudaMemcpy(prj, pcuPrj, sizeof(float) * nBatches * nu * nv * nview, cudaMemcpyDeviceToHost);
	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "cDistanceDrivenFanProjection failed: " << e.what()
			<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	if (pcuImg != NULL) cudaFree(pcuImg);
	if (pcuPrj != NULL) cudaFree(pcuPrj);
	if (pcuDeg != NULL) cudaFree(pcuDeg);

	return cudaGetLastError();

}



/*
preweighting for backpojection
It will weight the projections by the esitimated ray length
*/
__global__ void PreweightBPKernel(float* pPrjs, const float* pDeg, size_t nview, const Detector det, float dx, float dy)
{
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iv = blockIdx.y * blockDim.y + threadIdx.y;
	int iview = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iv >= det.nv || iview >= nview)
	{
		return;
	}

	float a = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float deg = pDeg[iview];

	if (fabsf(__cosf(deg)) > abs(sinf(deg)))
	{
		// Y as main axis
		pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] *= dy / fabsf(__cosf(deg - a));
	}
	else
	{
		//X as main axis
		pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] *= dx / fabsf(__sinf(deg - a));
	}
		
}

/*
preweighting for backpojection
It will weight the projections by the esitimated ray length
*/
__global__ void PreweightBPParallelKernel(
	float* pPrjs, const float* pDeg, size_t nview, const Detector det, float dx, float dy
)
{
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iv = blockIdx.y * blockDim.y + threadIdx.y;
	int iview = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iv >= det.nv || iview >= nview)
	{
		return;
	}

	float deg = pDeg[iview];

	if (fabsf(__cosf(deg)) > abs(sinf(deg)))
	{
		// Y as main axis
		pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] *= dy / fabsf(__cosf(deg));
	}
	else
	{
		//X as main axis
		pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] *= dx / fabsf(__sinf(deg));
	}
		
}

__device__ float GetProjectionOnDetector(float x, float y, float dsd, float dso, float cosDeg, float sinDeg)
{
	float rx =  x * cosDeg + y * sinDeg;
	float ry = -x * sinDeg + y * cosDeg;
	return atanf(rx / (ry + dso));

}

__device__ float GetProjectionOnDetectorParallel(float x, float y, float cosDeg, float sinDeg)
{
	return x * cosDeg + y * sinDeg;

}

__device__ float GetFBPWeight(float x, float y, float dsd, float dso, float cosDeg, float sinDeg)
{
	float rx =  x * cosDeg + y * sinDeg;
	float ry = -x * sinDeg + y * cosDeg;

	return dso*dso / (rx*rx + (dso+ry)*(dso+ry));

}

/*
Distance driven fanbeam backprojection

pImg - image of size (nz, ny, nx)
pAcc - accumulation of pojection along u, of size (nview, nv, nu+1)
pDeg - projection angles, size [nview]
nviews - total view number
grid - image grid
det - detector information
dsd - distance between source and detector
dso - distance between source and iso-center
isFBP - if use FBP weighting for backprojection

*/
__global__ void DDBPFanKernel(
	float* pImg,
	float* pAcc,
	const float* pDeg,
	size_t nview,
	const Grid grid,
	const Detector det,
	float dsd,
	float dso,
	bool isFBP
)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= grid.nx || iy >= grid.ny || iz >= grid.nz)
	{
		return;
	}

	float x = (ix - (grid.nx - 1) / 2.0f) * grid.dx;
	float x1 = x - grid.dx / 2.0f;
	float x2 = x + grid.dx / 2.0f;
	float y = (iy - (grid.ny - 1) / 2.0f) * grid.dy;
	float y1 = y - grid.dy / 2.0f;
	float y2 = y + grid.dy / 2.0f;
	float iv = (iz - (grid.nz - 1) / 2.0f) * grid.dz / det.dv + det.off_v + (det.nv - 1.0f) / 2.f;

	for (int iview = 0; iview < nview; iview++)
	{
		float deg = pDeg[iview];

		float cosDeg = __cosf(deg);
		float sinDeg = __sinf(deg);

		// For the u direction, the origin is the border of pixel 0
		// a1 and a2 are directly corresponding to the pixel-center coordinates of the accumulated pAcc
		float a1, a2;
		if (fabsf(cosDeg) > fabsf(sinDeg))
		{
			a1 = GetProjectionOnDetector(x1, y, dsd, dso, cosDeg, sinDeg);
			a2 = GetProjectionOnDetector(x2, y, dsd, dso, cosDeg, sinDeg);
		}
		else
		{
			a1 = GetProjectionOnDetector(x, y1, dsd, dso, cosDeg, sinDeg);
			a2 = GetProjectionOnDetector(x, y2, dsd, dso, cosDeg, sinDeg);
		}
		a1 = -(a1 / det.du + det.off_u) + det.nu / 2.0f;
		a2 = -(a2 / det.du + det.off_u) + det.nu / 2.0f;

		// make sure a1 < a2
		if (a1 > a2)
		{
			float t = a1;
			a1 = a2;
			a2 = t;
		}

		float val = (
			InterpolateXY(pAcc, a2, iv, iview, det.nu + 1, det.nv, nview)
			- InterpolateXY(pAcc, a1, iv, iview, det.nu + 1, det.nv, nview)
		) / (a2 - a1);

		if (isFBP)
		{
			pImg[iz * grid.nx * grid.ny + iy * grid.nx + ix] += val * GetFBPWeight(x, y, dsd, dso, cosDeg, sinDeg);
		}
		else
		{
			pImg[iz * grid.nx * grid.ny + iy * grid.nx + ix] += val;
		}

	}

}

/*
Distance driven parallel backprojection

pImg - image of size (nz, ny, nx)
pAcc - accumulation of pojection along u, of size (nview, nv, nu+1)
pDeg - projection angles, size [nview]
nviews - total view number
grid - image grid
det - detector information

*/
__global__ void DDBPParallelKernel(
	float* pImg,
	float* pAcc,
	const float* pDeg,
	size_t nview,
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

	float x = (ix - (grid.nx - 1) / 2.0f) * grid.dx;
	float x1 = x - grid.dx / 2.0f;
	float x2 = x + grid.dx / 2.0f;
	float y = (iy - (grid.ny - 1) / 2.0f) * grid.dy;
	float y1 = y - grid.dy / 2.0f;
	float y2 = y + grid.dy / 2.0f;
	float iv = (iz - (grid.nz - 1) / 2.0f) * grid.dz / det.dv + det.off_v + (det.nv - 1.0f) / 2.f;

	for (int iview = 0; iview < nview; iview++)
	{
		float deg = pDeg[iview];

		float cosDeg = __cosf(deg);
		float sinDeg = __sinf(deg);

		// For the u direction, the origin is the border of pixel 0
		// u1 and u2 are directly corresponding to the pixel-center coordinates of the accumulated pAcc
		float u1, u2;
		if (fabsf(cosDeg) > fabsf(sinDeg))
		{
			u1 = GetProjectionOnDetectorParallel(x1, y, cosDeg, sinDeg);
			u2 = GetProjectionOnDetectorParallel(x2, y, cosDeg, sinDeg);
		}
		else
		{
			u1 = GetProjectionOnDetectorParallel(x, y1, cosDeg, sinDeg);
			u2 = GetProjectionOnDetectorParallel(x, y2, cosDeg, sinDeg);
		}
		u1 = -(u1 / det.du + det.off_u) + det.nu / 2.0f;
		u2 = -(u2 / det.du + det.off_u) + det.nu / 2.0f;

		// make sure a1 < a2
		if (u1 > u2)
		{
			float t = u1;
			u1 = u2;
			u2 = t;
		}

		float val = (
			InterpolateXY(pAcc, u2, iv, iview, det.nu + 1, det.nv, nview)
			- InterpolateXY(pAcc, u1, iv, iview, det.nu + 1, det.nv, nview)
		) / (u2 - u1);

		pImg[iz * grid.nx * grid.ny + iy * grid.nx + ix] += val;

	}

}

void DistanceDrivenFan::Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg)
{
	bool isFBP = (typeProjector == 1);

	try
	{
		Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
		Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

		for (int ib = 0; ib < nBatches; ib++)
		{
			cudaMemcpyAsync(pWeightedPrjs, pcuPrj + ib * nu * nv * nview, sizeof(float) * nu * nv * nview, cudaMemcpyDeviceToDevice, m_stream);
			cudaStreamSynchronize(m_stream);

			// pre-weight for iterative BP to make it conjugate to FP
			dim3 threadsDet, blocksDet;
			if (!isFBP) // not FBP
			{
				GetThreadsForXZ(threadsDet, blocksDet, nu, nv, nview);
				PreweightBPKernel<<<blocksDet, threadsDet, 0, m_stream>>>(pWeightedPrjs, pcuDeg, nview, det, grid.dx, grid.dy);
				cudaStreamSynchronize(m_stream);
			}

			// step 1: accumulate projections along u axis
			dim3 threadU(1, 1, 64);
			dim3 blockU(1, nv, ceilf(nview / 64.0f));
			AccumulateKernelX<<<blockU, threadU, 0, m_stream>>>(pAccU, pWeightedPrjs, nu, nv, nview);
			cudaStreamSynchronize(m_stream);

			// step 2: backprojection
			dim3 threads, blocks;
			GetThreadsForXY(threads, blocks, nx, ny, nz);
			DDBPFanKernel<<<blocks, threads, 0, m_stream>>>(pcuImg + ib * nx * ny * nz, pAccU, pcuDeg, nview, grid, det, dsd, dso, isFBP);
			cudaStreamSynchronize(m_stream);
		}
	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "DistanceDrivenFan::Backprojection error: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}

}

void DistanceDrivenParallel::Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg)
{
	bool isFBP = (typeProjector == 1);

	try
	{
		Grid grid = MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz);
		Detector det = MakeDetector(nu, nv, du, dv, off_u, off_v);

		for (int ib = 0; ib < nBatches; ib++)
		{
			cudaMemcpyAsync(pWeightedPrjs, pcuPrj + ib * nu * nv * nview, sizeof(float) * nu * nv * nview, cudaMemcpyDeviceToDevice, m_stream);
			cudaStreamSynchronize(m_stream);

			// pre-weight for iterative BP to make it conjugate to FP
			dim3 threadsDet, blocksDet;
			if (!isFBP) // not FBP
			{
				GetThreadsForXZ(threadsDet, blocksDet, nu, nv, nview);
				PreweightBPParallelKernel<<<blocksDet, threadsDet, 0, m_stream>>>(pWeightedPrjs, pcuDeg, nview, det, grid.dx, grid.dy);
				cudaStreamSynchronize(m_stream);
			}

			// step 1: accumulate projections along u axis
			dim3 threadU(1, 1, 64);
			dim3 blockU(1, nv, ceilf(nview / 64.0f));
			AccumulateKernelX<<<blockU, threadU, 0, m_stream>>>(pAccU, pWeightedPrjs, nu, nv, nview);
			cudaStreamSynchronize(m_stream);

			// step 2: backprojection
			dim3 threads, blocks;
			GetThreadsForXY(threads, blocks, nx, ny, nz);
			DDBPParallelKernel<<<blocks, threads, 0, m_stream>>>(pcuImg + ib * nx * ny * nz, pAccU, pcuDeg, nview, grid, det);
			cudaStreamSynchronize(m_stream);
		}
	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "DistanceDrivenParallel::Backprojection error: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}
}

extern "C" int cupyDistanceDrivenFanBackprojection(
	float* img,
	const float* prj,
	const float* deg,
	size_t nBatches, 
	size_t nx,
	size_t ny,
	size_t nz,
	float dx,
	float dy,
	float dz,
	float cx,
	float cy,
	float cz,
	size_t nu,
	size_t nv,
	size_t nview,
	float da,
	float dv,
	float off_a,
	float off_v,
	float dsd,
	float dso,
	int typeProjector
)
{
	try
	{
		DistanceDrivenFan projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz, nu, nv, nview, da, dv, off_a, off_v, dsd, dso, typeProjector
		);
		
		projector.Allocate(false, true);
		projector.Backprojection(img, prj, deg);		
		projector.Free();
	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "cDistanceDrivenFanBackprojection failed: " << e.what()
			<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();

}


extern "C" int cupyDistanceDrivenParallelBackprojection(
	float* img,
	const float* prj,
	const float* deg,
	size_t nBatches, 
	size_t nx,
	size_t ny,
	size_t nz,
	float dx,
	float dy,
	float dz,
	float cx,
	float cy,
	float cz,
	size_t nu,
	size_t nv,
	size_t nview,
	float da,
	float dv,
	float off_a,
	float off_v,
	int typeProjector
)
{
	try
	{
		DistanceDrivenParallel projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz, nu, nv, nview, da, dv, off_a, off_v, 1000, 500, typeProjector
		);
		
		projector.Allocate(false, true);
		projector.Backprojection(img, prj, deg);		
		projector.Free();
	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "cDistanceDrivenParallelBackprojection failed: " << e.what()
			<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();

}

/*
extern "C" void cDistanceDrivenFanBackprojection(float* img, const float* prj, const float* deg,
		int nBatches, int nChannels, int nx, int ny, int nz, float dx, float dy, float dz,
		int nu, int nview, int nv, float da, float dv, float off_a, float off_v,
		float dsd, float dso, int typeProjector)
{
	float* pcuImg = NULL;
	float* pcuPrj = NULL;
	float* pcuDeg = NULL;
	try
	{
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nBatches * nx * ny * nz * nChannels))
		{
			throw runtime_error("pcuImg allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nu * nview * nv * nChannels))
		{
			throw runtime_error("pcuPrj allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuDeg, sizeof(float) * nview))
		{
			throw runtime_error("pcuDeg allocation failed");
		}

		cudaMemcpy(pcuDeg, deg, sizeof(float) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuPrj, prj, sizeof(float) * nBatches * nu * nview * nv * nChannels, cudaMemcpyHostToDevice);
		cudaMemset(pcuImg, 0, sizeof(float) * nBatches * nx * ny * nz * nChannels);
	}
	catch (exception &e)
	{
		if (pcuImg != NULL) cudaFree(pcuImg);
		if (pcuPrj != NULL) cudaFree(pcuPrj);
		if (pcuDeg != NULL) cudaFree(pcuDeg);

		ostringstream oss;
		oss << "cDistanceDrivenFanBackprojection failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw runtime_error(oss.str().c_str());
	}

	DistanceDrivenFan projector;
	projector.Setup(nBatches, nChannels, nx, ny, nz, dx, dy, dz,
			nu, nview, nv, da, dv, off_a, off_v, dsd, dso, typeProjector);

	projector.Backprojection(pcuImg, pcuPrj, pcuDeg);
	cudaMemcpy(img, pcuImg, sizeof(float) * nBatches * nx * ny * nz * nChannels, cudaMemcpyDeviceToHost);

	cudaFree(pcuImg);
	cudaFree(pcuPrj);
	cudaFree(pcuDeg);

}
*/