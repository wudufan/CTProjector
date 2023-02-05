#include "fbp.h"
#include "cudaMath.h"
#include "fbpMath.h"

#include <math.h>
#include <cufft.h>
#include <stdexcept>
#include <sstream>
#include <iostream>

using namespace std;


void fbpParallel::Filter(float* pcuFPrj, const float* pcuPrj)
{
	// the filter is carried out for each different v
	int filterLen = nu * 2 - 1;

	float* pcuPrjPad = NULL;                // projection
	cufftComplex* pcuFreqPrj = NULL;        // freq projection
	cufftComplex* pcuFilter = NULL;         // filter
	cufftHandle plan;                       // fft plan
    cufftHandle planInverse;

    try
    {
        if (cudaSuccess != cudaMalloc(&pcuPrjPad, sizeof(float) * filterLen * nview))
        {
            throw std::runtime_error("pcuPrjPad allocation failure in fbpParallel::Filter");
        }

        if (cudaSuccess != cudaMalloc(&pcuFreqPrj, sizeof(cufftComplex) * nu * nview))
        {
            throw std::runtime_error("pcuFreqPrj allocation failure in fbpParallel::Filter");
        }

        if (cudaSuccess != cudaMalloc(&pcuFilter, sizeof(cufftComplex) * nu * nview))
        {
            throw std::runtime_error("pcuFilter allocation failure in fbpParallel::Filter");
        }
        GetRamp(pcuFilter, nu, nview, du, typeProjector, m_stream, true);

        // no weighting for parallel filtering
        if (CUFFT_SUCCESS != cufftPlanMany(&plan, 1, &filterLen, NULL, 1, filterLen, NULL, 1, nu, CUFFT_R2C, nview))
        {
            throw std::runtime_error("fft plan error in fan3D::Filter");
        }
        cufftSetStream(plan, m_stream);

        if (CUFFT_SUCCESS != cufftPlanMany(&planInverse, 1, &filterLen, NULL, 1, nu, NULL, 1, filterLen, CUFFT_C2R, nview))
        {
            throw std::runtime_error("ifft plan error in fan3D::Filter");
        }
        cufftSetStream(plan, m_stream);

        // kernel threads and blocks
        dim3 threads(32, 32, 1);
        dim3 blocks(ceilf(nu / (float)threads.x), ceilf(nview / (float)threads.y), 1);
        float scale = PI / nview * du / filterLen;

        for (int ib = 0; ib < nBatches; ib++)
        {
            for (int iv = 0; iv < nv; iv++)
            {
                cudaMemsetAsync(pcuPrjPad, 0, sizeof(float) * filterLen * nview, m_stream);
                CopyPrjToPad<<<blocks, threads, 0, m_stream>>>(
                    pcuPrjPad, pcuPrj + ib * nu * nv * nview, iv, nu, filterLen, nv, nview
                );

                // no pre weighting for parallel filtering
                cudaDeviceSynchronize();

                cufftExecR2C(plan, pcuPrjPad, pcuFreqPrj);
                ComplexMultiply2D<<<blocks, threads, 0, m_stream>>>(pcuFreqPrj, pcuFreqPrj, pcuFilter, nu, nview);
                cudaDeviceSynchronize();
                cufftExecC2R(planInverse, pcuFreqPrj, pcuPrjPad);

                // post scaling
                Scale2D<<<blocks, threads, 0, m_stream>>>(
                    pcuPrjPad + nu - 1, pcuPrjPad + nu - 1, scale, nu, nview, filterLen, filterLen
                );

                CopyPadToPrj<<<blocks, threads, 0, m_stream>>>(
                    pcuPrjPad + nu - 1, pcuFPrj + ib * nu * nv * nview, iv, nu, filterLen, nv, nview
                );
            }
        }
    }
    catch(const std::exception& e)
    {   
        if (plan != 0) cufftDestroy(plan);
        if (planInverse != 0) cufftDestroy(planInverse);
        if (pcuPrjPad != NULL) cudaFree(pcuPrjPad);
        if (pcuFreqPrj != NULL) cudaFree(pcuFreqPrj);
        if (pcuFilter != NULL) cudaFree(pcuFilter);

        std::ostringstream oss;
        oss << "fbpParallel::Filter() failed: " << e.what()
            << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
        std::cerr << oss.str() << std::endl;
        throw std::runtime_error(oss.str().c_str());
    }

	cufftDestroy(plan);
	cufftDestroy(planInverse);
	cudaFree(pcuPrjPad);
	cudaFree(pcuFreqPrj);
	cudaFree(pcuFilter);
}

extern "C" int cupyfbpParallelFilter(
    float* pFPrj,
    const float* pPrj,
    int nBatches, 
    size_t nu,
    size_t nv,
    size_t nview,
    float du,
    float dv,
    float off_u,
    float off_v,
	int typeFilter = 0
) {
    fbpParallel filter;
	filter.Setup(
        nBatches, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        nu, nv, nview, du, dv, off_u, off_v, 0, 0, typeFilter
    );

    try
	{
        cudaMemset(pFPrj, 0, sizeof(float) * nBatches * nu * nv * nview);
		filter.Filter(pFPrj, pPrj);
	}
	catch (std::exception &e)
	{
		std::ostringstream oss;
		oss << "cFilterParallelFilter() failed: " << e.what()
            << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		std::cerr << oss.str() << std::endl;
	}

    return cudaGetLastError();
}

extern "C" int cfbpParallelFilter(
    float* pFPrj,
    const float* pPrj,
    int nBatches, 
    size_t nu,
    size_t nv,
    size_t nview,
    float du,
    float dv,
    float off_u,
    float off_v,
	int typeFilter = 0
)
{
	fbpParallel filter;
	filter.Setup(
        nBatches, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        nu, nv, nview, du, dv, off_u, off_v, 0, 0, typeFilter
    );
	float* pcuFPrj = NULL;
	float* pcuPrj = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuFPrj, sizeof(float) * nBatches * nu * nv * nview))
		{
			throw std::runtime_error("pcuFPrj allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nu * nv * nview))
		{
			throw std::runtime_error("pcuPrj allocation failed");
		}
		cudaMemset(pcuFPrj, 0, sizeof(float) * nBatches * nu * nv * nview);
		cudaMemcpy(pcuPrj, pPrj, sizeof(float) * nBatches * nu * nv * nview, cudaMemcpyHostToDevice);

		filter.Filter(pcuFPrj, pcuPrj);

        cudaMemcpy(pFPrj, pcuFPrj, sizeof(float) * nBatches * nu * nv * nview, cudaMemcpyDeviceToHost);
	}
	catch (std::exception &e)
	{
		std::ostringstream oss;
		oss << "cFilterParallelFilter() failed: " << e.what()
            << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		std::cerr << oss.str() << std::endl;
	}

	if (pcuFPrj != NULL) cudaFree(pcuFPrj);
    if (pcuPrj != NULL) cudaFree(pcuPrj);

    return cudaGetLastError();

}

const static int nzBatch = 5;
__global__ void bpParallelKernel3D(
    float* pImg,
    const float* prj,
    const float* pDeg,
    size_t nview,
    const Grid grid,
    const Detector det
)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int izBatch = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= grid.nx || iy >= grid.ny || izBatch * nzBatch >= grid.nz)
	{
		return;
	}

    // the image coordinates here has the lower left corner of the first pixel defined as (0,0,0)
    // because (ix,iy,iz) are based on the centers of the pixels, so an offset of 0.5 should be added. 
    register float3 pt = ImgToPhysics(make_float3(ix + 0.5f, iy + 0.5f, izBatch * nzBatch + 0.5f), grid);

	register float val[nzBatch] = {0};
	register float cosDeg, sinDeg, pu, pv;
	for (int iview = 0; iview < nview; iview++)
	{
		cosDeg = __cosf(pDeg[iview]);
		sinDeg = __sinf(pDeg[iview]);
		pu =  pt.x * cosDeg + pt.y * sinDeg;

		pu = -(pu / det.du + det.off_u) + (det.nu - 1.0f) / 2.0f;

#pragma unroll
		for (int iz = 0; iz < nzBatch; iz++)
		{
			pv = (pt.z + iz * grid.dz) / det.dv + det.off_v + (det.nv - 1.0f) / 2.f;

            val[iz] += InterpolateXY(prj, pu, pv, iview, det.nu, det.nv, nview, true);
		}

	}
#pragma unroll
	for (int iz = 0; iz < nzBatch; iz++)
	{
		if (iz + izBatch * nzBatch < grid.nz)
		{
			pImg[(iz + izBatch * nzBatch) * grid.nx * grid.ny + iy * grid.nx + ix] += val[iz];
		}
	}

}

void fbpParallel::Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg)
{
    dim3 threads(32, 16, 1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), ceilf(nz / (float)nzBatch));

	for (int ib = 0; ib < nBatches; ib++)
	{
        bpParallelKernel3D<<<blocks, threads, 0, m_stream>>>(
            pcuImg + ib * nx * ny * nz,
            pcuPrj + ib * nu * nv * nview,
            pcuDeg, 
            nview,
            MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz),
            MakeDetector(nu, nv, du, dv, off_u, off_v)
        );
        cudaDeviceSynchronize();
	}
}


extern "C" int cfbpParallelBackprojection(
    float* pImg,
    const float* pPrj,
    const float* pDeg,
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
    float off_v,
    int typeProjector = 0
)
{
    fbpParallel projector;
    projector.Setup(
        nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
        nu, nv, nview, du, dv, off_u, off_v, 1000, 500, typeProjector
    );

    float* pcuImg = NULL;
    float* pcuPrj = NULL;
    float* pcuDeg = NULL;
    try
    {
        if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nBatches * nx * ny * nz))
        {
            throw runtime_error("pcuImg allocation failed");
        }

        if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nu * nview * nv))
        {
            throw runtime_error("pcuPrj allocation failed");
        }

        if (cudaSuccess != cudaMalloc(&pcuDeg, sizeof(float) * nview))
        {
            throw runtime_error("pcuDeg allocation failed");
        }

        cudaMemcpy(pcuPrj, pPrj, sizeof(float) * nBatches * nu * nview * nv, cudaMemcpyHostToDevice);
        cudaMemcpy(pcuDeg, pDeg, sizeof(float) * nview, cudaMemcpyHostToDevice);
        cudaMemset(pcuImg, 0, sizeof(float) * nBatches * nx * ny * nz);

        projector.Backprojection(pcuImg, pcuPrj, pcuDeg);
        cudaMemcpy(pImg, pcuImg, sizeof(float) * nBatches * nx * ny * nz, cudaMemcpyDeviceToHost);

    }
    catch (exception &e)
    {
        ostringstream oss;
        oss << "cfbpParallelBackprojection() failed: " << e.what()
            << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
        cerr << oss.str() << endl;
    }

    if (pcuImg != NULL) cudaFree(pcuImg);
    if (pcuPrj != NULL) cudaFree(pcuPrj);
    if (pcuDeg != NULL) cudaFree(pcuDeg);

    return cudaGetLastError();
}
