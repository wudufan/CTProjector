#include "fbp.h"
#include "cudaMath.h"
#include "siddon.h"

#include <math.h>
#include <cufft.h>
#include <stdexcept>
#include <sstream>
#include <iostream>

using namespace std;

__global__ void ComplexMultiply2D(cufftComplex* res, const cufftComplex* op1, const cufftComplex* op2, size_t nx, size_t ny)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix >= nx || iy >= ny)
    {
        return;
    }

    size_t ind = iy * nx + ix;
    cufftComplex val1 = op1[ind];
    cufftComplex val2 = op2[ind];
    cufftComplex val;

    val.x = val1.x * val2.x - val1.y * val2.y;
    val.y = val1.x * val2.y + val1.y * val2.x;

    res[ind] = val;
}

// The pcuPrjPad is of size [nview, nuPad]
__global__ void CopyPrjToPad(float* pcuPrjPad, const float* pcuPrj, int iv, 
    size_t nu, size_t nuPad, size_t nv, size_t nview)
{
    int iu = blockIdx.x * blockDim.x + threadIdx.x;
    int iview = blockIdx.y * blockDim.y + threadIdx.y;

    if (iu >= nu || iview >= nview)
    {
        return;
    }

    pcuPrjPad[iview * nuPad + iu] = pcuPrj[iview * nu * nv + iv * nu + iu];
}

__global__ void CopyPadToPrj(const float* pcuPrjPad, float* pcuPrj, int iv,
    size_t nu, size_t nuPad, size_t nv, size_t nview)
{
    int iu = blockIdx.x * blockDim.x + threadIdx.x;
    int iview = blockIdx.y * blockDim.y + threadIdx.y;

    if (iu >= nu || iview >= nview)
    {
        return;
    }

    pcuPrj[iview * nu * nv + iv * nu + iu] = pcuPrjPad[iview * nuPad + iu];
}

void GetRamp(cufftComplex* pcuFreqKernel, size_t nu, size_t nview, float da, int filterType, cudaStream_t& stream,
    bool isEqualSpace = false)
{
    int filterLen = 2 * nu - 1;

    // fft plan
    cufftHandle plan = 0;
    cufftHandle planR2C = 0;
    cufftComplex* pcuRamp = NULL;
    cufftReal *pcuRealKernel = NULL;
    cufftComplex* pRamp = NULL;
    cufftComplex* pWindow = NULL;
    cufftReal* pRealKernel = NULL;

    try
    {
        if (CUFFT_SUCCESS != cufftPlan1d(&plan, filterLen, CUFFT_C2C, 1))
        {
            throw std::runtime_error("cufftPlan1d failure in GetRamp()");
        }
    
        if (CUFFT_SUCCESS != cufftSetStream(plan, stream))
        {
            throw std::runtime_error("cudaSetStream failure in GetRamp()");
        }
    
        // RL kernel
        if (cudaSuccess != cudaMalloc(&pcuRamp, sizeof(cufftComplex) * filterLen))
        {
            throw std::runtime_error("pcuRamp allocation error in GetRamp()");
        }
        pRamp = new cufftComplex [filterLen];
        if (isEqualSpace)
        {
            // equispace
            for (int i = 0; i < filterLen; i++)
            {
                int k = i - (nu - 1);
                if (k == 0)
                {
                    pRamp[i].x = 1 / (4 * da * da);
                }
                else if (k % 2 != 0)
                {
                    pRamp[i].x = -1 / (PI * PI * k * k * da * da);
                }
                else
                {
                    pRamp[i].x = 0;
                }
                pRamp[i].y = 0;
            }
        }
        else
        {
            // equiangular
            for (int i = 0; i < filterLen; i++)
            {
                int k = i - (nu - 1);
                if (k == 0)
                {
                    pRamp[i].x = 1 / (4 * da * da);
                }
                else if (k % 2 != 0)
                {
                    pRamp[i].x = -1 / (PI * PI * sinf(k*da) * sinf(k*da));
                }
                else
                {
                    pRamp[i].x = 0;
                }
                pRamp[i].y = 0;
            }
        }
    
        cudaMemcpyAsync(pcuRamp, pRamp, sizeof(cufftComplex) * filterLen, cudaMemcpyHostToDevice, stream);
        cufftExecC2C(plan, pcuRamp, pcuRamp, CUFFT_FORWARD);
        cudaMemcpyAsync(pRamp, pcuRamp, sizeof(cufftComplex) * filterLen, cudaMemcpyDeviceToHost, stream);
    
        // weighting window in frequency domain
        pWindow = new cufftComplex [filterLen];
        switch(filterType)
        {
        case FILTER_HAMMING:
            // Hamming
            for (int i = 0; i < filterLen; i++)
            {
                pWindow[i].x = 0.54f + 0.46f * cosf(2 * PI * i / (float)(filterLen-1));
                pWindow[i].y = 0;
            }
            break;
        case FILTER_HANN:
            for (int i = 0; i < filterLen; i++)
            {
                pWindow[i].x = 0.5f + 0.5f * cosf(2 * PI * i / (float)(filterLen-1));
                pWindow[i].y = 0;
            }
            break;
        case FILTER_COSINE:
            for (int i = 0; i < filterLen; i++)
            {
                pWindow[i].x = abs(cosf(PI * i / (float)(filterLen-1)));
                pWindow[i].y = 0;
            }
            break;
        default:
            for (int i = 0; i < filterLen; i++)
            {
                pWindow[i].x = 1;
                pWindow[i].y = 0;
            }
        }
    
        // Apply window on the filter
        for (int i = 0; i < filterLen; i++)
        {
            float real = pRamp[i].x * pWindow[i].x - pRamp[i].y * pWindow[i].y;
            float imag = pRamp[i].x * pWindow[i].y + pRamp[i].y * pWindow[i].x;
            pRamp[i].x = real;
            pRamp[i].y = imag;
        }
    
        cudaMemcpyAsync(pcuRamp, pRamp, sizeof(cufftComplex) * filterLen, cudaMemcpyHostToDevice, stream);
        cufftExecC2C(plan, pcuRamp, pcuRamp, CUFFT_INVERSE);
        cudaMemcpyAsync(pRamp, pcuRamp, sizeof(cufftComplex) * filterLen, cudaMemcpyDeviceToHost, stream);

        if (cudaSuccess != cudaMalloc(&pcuRealKernel, sizeof(cufftReal) * filterLen))
        {
            throw std::runtime_error("pRealKernel allocation error in getRampEA");
        }
        cufftReal* pRealKernel = new cufftReal [filterLen];
        for (int i = 0; i < filterLen; i++)
        {
            pRealKernel[i] = pRamp[i].x / filterLen;
        }
        cudaMemcpyAsync(pcuRealKernel, pRealKernel, sizeof(cufftReal) * filterLen, cudaMemcpyHostToDevice, stream);
    
        cufftHandle planR2C;
        cufftPlan1d(&planR2C, filterLen, CUFFT_R2C, 1);
        cufftSetStream(planR2C, stream);
        cufftExecR2C(planR2C, pcuRealKernel, pcuFreqKernel);
        for (int i = 1; i < nview; i++)
        {
            cudaMemcpyAsync(pcuFreqKernel + i * nu, pcuFreqKernel, sizeof(cufftComplex) * nu, cudaMemcpyDeviceToDevice, stream);
        }

    }
    catch (exception &e)
    {
        if (pRamp != NULL) delete [] pRamp;
        if (pWindow != NULL) delete [] pWindow;
        if (pRealKernel != NULL) delete [] pRealKernel;
        if (pcuRamp != NULL) cudaFree(pcuRamp);
        if (pcuRealKernel != NULL) cudaFree(pcuRealKernel);
        if (plan != 0) cufftDestroy(plan);
        if (planR2C != 0) cufftDestroy(planR2C);

        ostringstream oss;
        oss << "GetRamp() failed: " << e.what()
                << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
        cerr << oss.str() << endl;
        throw runtime_error(oss.str().c_str());
    }

    if (pRamp != NULL) delete [] pRamp;
    if (pWindow != NULL) delete [] pWindow;
    if (pRealKernel != NULL) delete [] pRealKernel;
    if (pcuRamp != NULL) cudaFree(pcuRamp);
    if (pcuRealKernel != NULL) cudaFree(pcuRealKernel);
    if (plan != 0) cufftDestroy(plan);
    if (planR2C != 0) cufftDestroy(planR2C);

}

void fbpFan::Filter(float* pcuFPrj, const float* pcuPrj)
{
    bool isEqualSpace = false;

    // the filter is carried out for each different v
    int filterLen = nu * 2 - 1;
    float* pcuPrjPad = NULL;
    cufftComplex* pcuFreqPrj = NULL;
    cufftComplex* pcuFilter = NULL;
    float* pw = NULL;
    float* pcuw = NULL;
    cufftHandle plan = 0;
    cufftHandle planInverse = 0;

    try
    {
        // projection
        if (cudaSuccess != cudaMalloc(&pcuPrjPad, sizeof(float) * filterLen * nview))
        {
            throw std::runtime_error("pcuPrjPad allocation failure in fan3D::Filter");
        }

        // freq projection
        if (cudaSuccess != cudaMalloc(&pcuFreqPrj, sizeof(cufftComplex) * nu * nview))
        {
            throw std::runtime_error("pcuFreqPrj allocation failure in fan3D::Filter");
        }

        // filter
        if (cudaSuccess != cudaMalloc(&pcuFilter, sizeof(cufftComplex) * nu * nview))
        {
            throw std::runtime_error("pcuFilter allocation failure in fan3D::Filter");
        }
        GetRamp(pcuFilter, nu, nview, du, typeProjector, m_stream, isEqualSpace);

        // Get projection weighting
        pw = new float [nu];
        if (isEqualSpace)
        {
            for (int i = 0; i < nu; i++)
            {
                float u = ((i - (nu - 1) / 2.f) - off_u) * du;
                pw[i] = dsd / sqrtf(dsd * dsd + u * u);
            }
        }
        else
        {
            for (int i = 0; i < nu; i++)
            {
                float angle = ((i - (nu - 1) / 2.f) - off_u) * du;
                pw[i] = cosf(angle);
            }
        }

        
        if (cudaSuccess != cudaMalloc(&pcuw, sizeof(float) * nu * nview))
        {
            throw std::runtime_error("pcuw allocation failure in fan3D::Filter");
        }
        cudaMemcpyAsync(pcuw, pw, sizeof(float) * nu, cudaMemcpyHostToDevice, m_stream);
        for (int i = 1; i < nview; i++)
        {
            cudaMemcpyAsync(pcuw + i * nu, pcuw, sizeof(float) * nu, cudaMemcpyDeviceToDevice, m_stream);
        }

        // fft plan
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
        dim3 threads(32, 16, 1);
        dim3 blocks(ceilf(nu / (float)threads.x), ceilf(nview / (float)threads.y), 1);
        float scale;
        if (isEqualSpace)
        {
            scale = PI / nview * du * dsd / dso / filterLen;
        }
        else
        {
            scale = PI / nview * du / dso / filterLen;
        }

        for (int ib = 0; ib < nBatches; ib++)
        {
            for (int iv = 0; iv < nv; iv++)
            {
                cudaMemsetAsync(pcuPrjPad, 0, sizeof(float) * filterLen * nview, m_stream);
                CopyPrjToPad<<<blocks, threads, 0, m_stream>>>(pcuPrjPad,
                        pcuPrj + ib * nu * nv * nview,
                        iv, nu, filterLen, nv, nview);

                // pre weighting
                Multiply2D<<<blocks, threads, 0, m_stream>>>(pcuPrjPad, pcuPrjPad, pcuw,
                        nu, nview, filterLen, filterLen, nu);
                cudaDeviceSynchronize();

                cufftExecR2C(plan, pcuPrjPad, pcuFreqPrj);
                ComplexMultiply2D<<<blocks, threads, 0, m_stream>>>(pcuFreqPrj, pcuFreqPrj, pcuFilter, nu, nview);
                cudaDeviceSynchronize();
                cufftExecC2R(planInverse, pcuFreqPrj, pcuPrjPad);

                // post scaling
                Scale2D<<<blocks, threads, 0, m_stream>>>(pcuPrjPad + nu - 1, pcuPrjPad + nu - 1,
                        scale, nu, nview, filterLen, filterLen);

                CopyPadToPrj<<<blocks, threads, 0, m_stream>>>(pcuPrjPad + nu - 1,
                        pcuFPrj + ib * nu * nv * nview,
                        iv, nu, filterLen, nv, nview);
            }
        }
    }
    catch (exception &e)
    {
        if (plan != 0) cufftDestroy(plan);
        if (planInverse != 0) cufftDestroy(planInverse);
        if (pcuPrjPad != NULL) cudaFree(pcuPrjPad);
        if (pcuFreqPrj != NULL) cudaFree(pcuFreqPrj);
        if (pcuFilter != NULL) cudaFree(pcuFilter);
        if (pcuw != NULL) cudaFree(pcuw);
        if (pw != NULL) delete [] pw;

        ostringstream oss;
        oss << "fbpFan::Filter() failed: " << e.what()
                << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
        cerr << oss.str() << endl;
        throw runtime_error(oss.str().c_str());
    }

    if (plan != 0) cufftDestroy(plan);
    if (planInverse != 0) cufftDestroy(planInverse);
    if (pcuPrjPad != NULL) cudaFree(pcuPrjPad);
    if (pcuFreqPrj != NULL) cudaFree(pcuFreqPrj);
    if (pcuFilter != NULL) cudaFree(pcuFilter);
    if (pcuw != NULL) cudaFree(pcuw);
    if (pw != NULL) delete [] pw;

}

extern "C" int cfbpFanFilter(float* pFPrj, const float* pPrj,
    int nBatches, 
    size_t nu, size_t nv, size_t nview, float da, float dv, float off_a, float off_v,
    float dsd, float dso, int typeFilter = 0)
{
    fbpFan filter;
    filter.Setup(nBatches, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        nu, nv, nview, da, dv, off_a, off_v, dsd, dso, typeFilter);
    float* pcuFPrj = NULL;
    float* pcuPrj = NULL;

    try
    {
        if (cudaSuccess != cudaMalloc(&pcuFPrj, sizeof(float) * nBatches * nu * nv * nview))
        {
            throw runtime_error("pcuFPrj allocation failed");
        }
        if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nu * nv * nview))
        {
            throw runtime_error("pcuPrj allocation failed");
        }
        cudaMemset(pcuFPrj, 0, sizeof(float) * nBatches * nu * nv * nview);
        cudaMemcpy(pcuPrj, pPrj, sizeof(float) * nBatches * nu * nv * nview, cudaMemcpyHostToDevice);

        filter.Filter(pcuFPrj, pcuPrj);

        cudaMemcpy(pFPrj, pcuFPrj, sizeof(float) * nBatches * nu * nv * nview, cudaMemcpyDeviceToHost);
    }
    catch (exception &e)
    {
        ostringstream oss;
        oss << "cFilterFanFilter() failed: " << e.what()
                << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
        cerr << oss.str() << endl;
    }

    if (pcuFPrj != NULL) cudaFree(pcuFPrj);
    if (pcuPrj != NULL) cudaFree(pcuPrj);

    return cudaGetLastError();

}

/**************************
Backprojection
**************************/

const static int nzBatch = 5;
__global__ void bpFanKernel3D(float* pImg, const float* prj, const float* pDeg,
    size_t nview, const Grid grid, const Detector det, float dsd, float dso, bool isFBP)
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
	register float cosDeg, sinDeg, rx, ry, pu, pv, a, dist;
	for (int iview = 0; iview < nview; iview++)
	{
		cosDeg = __cosf(pDeg[iview]);
		sinDeg = __sinf(pDeg[iview]);
		rx =  pt.x*cosDeg + pt.y*sinDeg;
		ry = -pt.x*sinDeg + pt.y*cosDeg;
		a = atanf(rx/(ry+dso));
		if (isFBP)
		{
			dist = dso*dso / (rx*rx + (dso+ry)*(dso+ry));
		}
		else
		{
			float sin_a = fabs(__sinf(a));
			if (sin_a > 1e-6f)
			{
				dist = fminf(grid.dy / __cosf(a), grid.dx / sin_a);
			}
			else
			{
				dist = grid.dy / __cosf(a);
			}
		}

		pu = -(a/det.du + det.off_u) + (det.nu - 1.0f) / 2.0f;

#pragma unroll
		for (int iz = 0; iz < nzBatch; iz++)
		{
			pv = (pt.z + iz * grid.dz) / det.dv + det.off_v + (det.nv - 1.0f) / 2.f;

            // val[iz] = dist;
            val[iz] += InterpolateXY(prj, pu, pv, iview, det.nu, det.nv, nview, true) * dist;
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

void fbpFan::Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg)
{
    dim3 threads(32, 16, 1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), ceilf(nz / (float)nzBatch));

	for (int ib = 0; ib < nBatches; ib++)
	{
        bpFanKernel3D<<<blocks, threads, 0, m_stream>>>(
            pcuImg + ib * nx * ny * nz, pcuPrj + ib * nu * nv * nview, pcuDeg, 
            nview,
            MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz),
            MakeDetector(nu, nv, du, dv, off_u, off_v),
            dsd, dso, true);
        cudaDeviceSynchronize();
	}
}

extern "C" int cfbpFanBackprojection(float* pImg, const float* pPrj, const float* pDeg,
    size_t nBatches, 
    size_t nx, size_t ny, size_t nz, float dx, float dy, float dz, float cx, float cy, float cz, 
    size_t nu, size_t nv, size_t nview, float da, float dv, float off_a, float off_v,
    float dsd, float dso)
{
    fbpFan projector;
    projector.Setup(nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
            nu, nv, nview, da, dv, off_a, off_v, dsd, dso);

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
        oss << "cfbpFanBackprojection() failed: " << e.what()
                << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
        cerr << oss.str() << endl;
    }

    if (pcuImg != NULL) cudaFree(pcuImg);
    if (pcuPrj != NULL) cudaFree(pcuPrj);
    if (pcuDeg != NULL) cudaFree(pcuDeg);

    return cudaGetLastError();
}