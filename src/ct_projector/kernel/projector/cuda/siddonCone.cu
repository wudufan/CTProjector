#include "siddon.h"
#include "projector.h"
#include "cudaMath.h"
#include "siddonCone.h"

#include <stdexcept>
#include <iostream>
#include <sstream>

using namespace std;

__global__ void SiddonConeProjectionArbitraryKernel(
	float* pPrj,
	const float* pImg,
	const float3* pDetCenter,
	const float3* pDetU,
	const float3* pDetV,
	const float3* pSrc,
	int nview,
	const Detector det,
	const Grid grid
)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iv = blockDim.y * blockIdx.y + threadIdx.y;
	int iview = blockDim.z * blockIdx.z + threadIdx.z;

	if (iu >= det.nu || iv >= det.nv || iview >= nview)
	{
		return;
	}

	float u = (iu - det.off_u - (det.nu - 1) / 2.0f) * det.du;
	float v = (iv - det.off_v - (det.nv - 1) / 2.0f) * det.dv;

	float3 src = pSrc[iview];
	float3 dst = UVToCart(u, v, pDetCenter[iview], pDetU[iview], pDetV[iview]);
	MoveSourceDstNearGrid(src, dst, grid);

	SiddonRayTracing(pPrj + iview * det.nu * det.nv + iv * det.nu + iu, pImg, src, dst, grid);

}

__global__ void SiddonConeBackprojectionArbitraryKernel(
	float* pImg,
	const float* pPrj,
	const float3* pDetCenter,
	const float3* pDetU,
	const float3* pDetV,
	const float3* pSrc,
	int nview,
	const Detector det,
	const Grid grid
)
{
	int iu = blockDim.x * blockIdx.x + threadIdx.x;
	int iv = blockDim.y * blockIdx.y + threadIdx.y;
	int iview = blockDim.z * blockIdx.z + threadIdx.z;

	if (iu >= det.nu || iv >= det.nv || iview >= nview)
	{
		return;
	}

	float u = (iu - det.off_u - (det.nu - 1) / 2.0f) * det.du;
	float v = (iv - det.off_v - (det.nv - 1) / 2.0f) * det.dv;

	float3 src = pSrc[iview];
	float3 dst = UVToCart(u, v, pDetCenter[iview], pDetU[iview], pDetV[iview]);
	MoveSourceDstNearGrid(src, dst, grid);

	SiddonRayTracingTransposeAtomicAdd(pImg, pPrj[iview * det.nu * det.nv + iv * det.nu + iu], src, dst, grid);

}

void SiddonCone::ProjectionArbitrary(
	const float* pcuImg,
	float* pcuPrj,
	const float3* pcuDetCenter,
	const float3* pcuDetU,
	const float3* pcuDetV,
	const float3* pcuSrc
)
{
	dim3 threads, blocks;
	GetThreadsForXY(threads, blocks, nu, nv, nview);

	for (int ib = 0; ib < nBatches; ib++)
	{
		SiddonConeProjectionArbitraryKernel<<<blocks, threads, 0, m_stream>>>(
			pcuPrj + ib * nu * nv * nview,
			pcuImg + ib * nx * ny * nz,
			pcuDetCenter, 
			pcuDetU,
			pcuDetV, 
			pcuSrc,
			nview,
			MakeDetector(nu, nv, du, dv, off_u, off_v),
			MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz)
		);

		cudaStreamSynchronize(m_stream);
	}

}

void SiddonCone::BackprojectionArbitrary(
	float* pcuImg,
	const float* pcuPrj,
	const float3* pcuDetCenter,
	const float3* pcuDetU,
	const float3* pcuDetV,
	const float3* pcuSrc
)
{
	dim3 threads, blocks;
	GetThreadsForXY(threads, blocks, nu, nv, nview);

	for (int ib = 0; ib < nBatches; ib++)
	{
		SiddonConeBackprojectionArbitraryKernel<<<blocks, threads, 0, m_stream>>>(
			pcuImg + ib * nx * ny * nz,
			pcuPrj + ib * nu * nv * nview,
			pcuDetCenter, 
			pcuDetU,
			pcuDetV, 
			pcuSrc,
			nview, 
			MakeDetector(nu, nv, du, dv, off_u, off_v),
			MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz)
		);

		cudaStreamSynchronize(m_stream);
	}
}

extern "C" int cSiddonConeProjectionArbitrary(
	float* prj,
	const float* img,
	const float* detCenter,
	const float* detU,
	const float* detV,
	const float* src,
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
	float du,
	float dv,
	float off_u,
	float off_v
)
{
	float* pcuPrj = NULL;
	float* pcuImg = NULL;
	float3* pcuDetCenter = NULL;
	float3* pcuDetU = NULL;
	float3* pcuDetV = NULL;
	float3* pcuSrc = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nu * nv * nview * nBatches))
		{
			throw ("pcuPrj allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nx * ny * nz * nBatches))
		{
			throw ("pcuImg allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetCenter, sizeof(float3) * nview))
		{
			throw ("pcuDetCenter allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetU, sizeof(float3) * nview))
		{
			throw ("pcuDetU allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetV, sizeof(float3) * nview))
		{
			throw ("pcuDetV allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuSrc, sizeof(float3) * nview))
		{
			throw ("pcuSrc allocation failed");
		}

		cudaMemcpy(pcuImg, img, sizeof(float) * nx * ny * nz * nBatches, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDetCenter, detCenter, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDetU, detU, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDetV, detV, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuSrc, src, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemset(pcuPrj, 0, sizeof(float) * nu * nview * nv * nBatches);

		SiddonCone projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, du, dv, off_u, off_v, 0, 0, 0
		);

		projector.ProjectionArbitrary(pcuImg, pcuPrj, pcuDetCenter, pcuDetU, pcuDetV, pcuSrc);
		cudaMemcpy(prj, pcuPrj, sizeof(float) * nu * nv * nview * nBatches, cudaMemcpyDeviceToHost);
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cSiddonConeProjectionArbitrary() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;

	}

	if (pcuPrj != NULL) cudaFree(pcuPrj);
	if (pcuImg != NULL) cudaFree(pcuImg);
	if (pcuDetCenter != NULL) cudaFree(pcuDetCenter);
	if (pcuDetU != NULL) cudaFree(pcuDetU);
	if (pcuDetV != NULL) cudaFree(pcuDetV);
	if (pcuSrc != NULL) cudaFree(pcuSrc);

	return cudaGetLastError();

}

extern "C" int cupySiddonConeProjectionArbitrary(
	float* prj,
	const float* img,
	const float* detCenter,
	const float* detU,
	const float* detV,
	const float* src,
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
	float du,
	float dv,
	float off_u,
	float off_v
)
{
	try
	{
		SiddonCone projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, du, dv, off_u, off_v, 0, 0, 0
		);

		projector.ProjectionArbitrary(img, prj, (const float3*)detCenter, (const float3*)detU, (const float3*)detV, (const float3*)src);
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cSiddonConeProjectionArbitrary() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;

	}

	return cudaGetLastError();

}

extern "C" int cSiddonConeBackprojectionArbitrary(
	float* img,
	const float* prj,
	const float* detCenter,
	const float* detU,
	const float* detV,
	const float* src,
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
	float du,
	float dv,
	float off_u,
	float off_v
)
{
	float* pcuPrj = NULL;
	float* pcuImg = NULL;
	float3* pcuDetCenter = NULL;
	float3* pcuDetU = NULL;
	float3* pcuDetV = NULL;
	float3* pcuSrc = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nu * nv * nview * nBatches))
		{
			throw ("pcuPrj allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nx * ny * nz * nBatches))
		{
			throw ("pcuImg allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetCenter, sizeof(float3) * nview))
		{
			throw ("pcuDetCenter allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetU, sizeof(float3) * nview))
		{
			throw ("pcuDetU allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetV, sizeof(float3) * nview))
		{
			throw ("pcuDetV allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuSrc, sizeof(float3) * nview))
		{
			throw ("pcuSrc allocation failed");
		}

		cudaMemcpy(pcuPrj, prj, sizeof(float) * nu * nv * nview * nBatches, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDetCenter, detCenter, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDetU, detU, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDetV, detV, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuSrc, src, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemset(pcuImg, 0, sizeof(float) * nx * ny * nz * nBatches);

		SiddonCone projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, du, dv, off_u, off_v, 0, 0, 0
		);

		projector.BackprojectionArbitrary(pcuImg, pcuPrj, pcuDetCenter, pcuDetU, pcuDetV, pcuSrc);

		cudaMemcpy(img, pcuImg, sizeof(float) * nx * ny * nz * nBatches, cudaMemcpyDeviceToHost);
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cSiddonConeBackprojectionArbitrary() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	if (pcuPrj != NULL) cudaFree(pcuPrj);
	if (pcuImg != NULL) cudaFree(pcuImg);
	if (pcuDetCenter != NULL) cudaFree(pcuDetCenter);
	if (pcuDetU != NULL) cudaFree(pcuDetU);
	if (pcuDetV != NULL) cudaFree(pcuDetV);
	if (pcuSrc != NULL) cudaFree(pcuSrc);

	return cudaGetLastError();

}

extern "C" int cupySiddonConeBackprojectionArbitrary(
	float* img,
	const float* prj,
	const float* detCenter,
	const float* detU,
	const float* detV,
	const float* src,
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
	float du,
	float dv,
	float off_u,
	float off_v
)
{
	try
	{
		SiddonCone projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, du, dv, off_u, off_v, 0, 0, 0
		);

		projector.BackprojectionArbitrary(img, prj, (const float3*)detCenter, (const float3*)detU, (const float3*)detV, (const float3*)src);
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cSiddonConeBackprojectionArbitrary() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();

}
