#include "siddonFan.h"
#include "siddon.h"
#include "cudaMath.h"

#include <math.h>
#include <stdexcept>
#include <iostream>
#include <sstream>

using namespace std;

__global__ void SiddonFanProjectionKernel(
    float* pPrj,
    const float* pImg,
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

	if (iu >= det.nu || iview >= nview || iv >= det.nv)
	{
		return;
	}

	float cosDeg = __cosf(pDeg[iview]);
	float sinDeg = __sinf(pDeg[iview]);
	float a = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float sin_a = __sinf(a);
	float cos_a = __cosf(a);
	float z = (iv - (det.nv-1) / 2.0f + det.off_v) * det.dv;

	// src, dst points, and convert to image coordinate
	float3 src = make_float3(dso * sinDeg, -dso * cosDeg, z);
	float3 dstRel = make_float3(dsd * sin_a, -dso + dsd * cos_a, z);
	float3 dst = make_float3(dstRel.x * cosDeg - dstRel.y * sinDeg, dstRel.x * sinDeg + dstRel.y * cosDeg, z);

	SiddonRayTracing(pPrj + iview * det.nu * det.nv + iv * det.nu + iu, pImg, src, dst, grid);

}

__global__ void SiddonFanBackprojectionKernel(
    float* pImg,
    const float* pPrj,
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

	if (iu >= det.nu || iview >= nview || iv >= det.nv)
	{
		return;
	}

	float cosDeg = __cosf(pDeg[iview]);
	float sinDeg = __sinf(pDeg[iview]);
	float a = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float sin_a = __sinf(a);
	float cos_a = __cosf(a);
	float z = (iv - (det.nv-1) / 2.0f + det.off_v) * det.dv;

	// src, dst points, and convert to image coordinate
	float3 src = make_float3(dso * sinDeg, -dso * cosDeg, z);
	float3 dstRel = make_float3(dsd * sin_a, -dso + dsd * cos_a, z);
	float3 dst = make_float3(dstRel.x * cosDeg - dstRel.y * sinDeg, dstRel.x * sinDeg + dstRel.y * cosDeg, z);

	SiddonRayTracingTransposeAtomicAdd(pImg, pPrj[iview * det.nu * det.nv + iv * det.nu + iu], src, dst, grid);

}

void SiddonFan::Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg)
{
	dim3 threads, blocks;
	GetThreadsForXZ(threads, blocks, nu, nv, nview);

	for (int ib = 0; ib < nBatches; ib++)
	{
        SiddonFanProjectionKernel<<<blocks, threads, 0, m_stream>>>(
            pcuPrj + ib * nu * nv * nview, 
            pcuImg + ib * nx * ny * nz, 
            pcuDeg,
            nview,
            MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz),
            MakeDetector(nu, nv, du, dv, off_u, off_v), 
            dsd,
            dso
        );
        cudaDeviceSynchronize();
	}

}

void SiddonFan::Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg)
{
	dim3 threads, blocks;
	GetThreadsForXZ(threads, blocks, nu, nv, nview);

	for (int ib = 0; ib < nBatches; ib++)
	{
        SiddonFanBackprojectionKernel<<<blocks, threads, 0, m_stream>>>(
            pcuImg + ib * nx * ny * nz,
            pcuPrj + ib * nu * nview * nv, 
            pcuDeg,
            nview,
            MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz),
            MakeDetector(nu, nv, du, dv, off_u, off_v), 
            dsd,
            dso
        );
        cudaDeviceSynchronize();
	}

}

extern "C" int cSiddonFanProjection(
    float* prj,
    const float* img,
    const float* deg,
    int nBatches,
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

        SiddonFan projector;
        projector.Setup(
            nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
            nu, nv, nview, da, dv, off_a, off_v, dsd, dso
        );

        projector.Projection(pcuImg, pcuPrj, pcuDeg);
        cudaMemcpy(prj, pcuPrj, sizeof(float) * nBatches * nu * nview * nv, cudaMemcpyDeviceToHost);
    }
    catch (exception &e)
    {
        ostringstream oss;
        oss << "cSiddonFanProjection failed: " << e.what()
            << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
        cerr << oss.str() << endl;
    }

    if (pcuImg != NULL) cudaFree(pcuImg);
    if (pcuPrj != NULL) cudaFree(pcuPrj);
    if (pcuDeg != NULL) cudaFree(pcuDeg);

    return cudaGetLastError();

}

extern "C" int cSiddonFanBackprojection(
    float* img,
    const float* prj,
    const float* deg,
    int nBatches,
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
        cudaMemcpy(pcuPrj, prj, sizeof(float) * nBatches * nu * nv * nview, cudaMemcpyHostToDevice);
        cudaMemset(pcuImg, 0, sizeof(float) * nBatches * nx * ny * nz);

        SiddonFan projector;
        projector.Setup(
            nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
            nu, nv, nview, da, dv, off_a, off_v, dsd, dso
        );

        projector.Backprojection(pcuImg, pcuPrj, pcuDeg);
        cudaMemcpy(img, pcuImg, sizeof(float) * nBatches * nx * ny * nz, cudaMemcpyDeviceToHost);

    }
    catch (exception &e)
    {
        ostringstream oss;
        oss << "cSiddonFanBackprojection failed: " << e.what()
            << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
        cerr << oss.str() << endl;
    }

    if (pcuImg != NULL) cudaFree(pcuImg);
    if (pcuPrj != NULL) cudaFree(pcuPrj);
    if (pcuDeg != NULL) cudaFree(pcuDeg);

    return cudaGetLastError();

}