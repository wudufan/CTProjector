#include "fbpMath.h"
#include "fbp.h"

#include <math.h>
#include <cufft.h>
#include <stdexcept>
#include <sstream>
#include <iostream>

using namespace std;

__global__ void ComplexMultiply2D(
    cufftComplex* res, const cufftComplex* op1, const cufftComplex* op2, size_t nx, size_t ny
)
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

__global__ void FilterByFreqMultiply1D(
    cufftComplex* res, const cufftComplex* src, const cufftComplex* filter, size_t nx, size_t ny
)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix >= nx || iy >= ny)
    {
        return;
    }

    size_t ind = iy * nx + ix;
    cufftComplex val1 = src[ind];
    cufftComplex val2 = filter[ix];
    cufftComplex val;

    val.x = val1.x * val2.x - val1.y * val2.y;
    val.y = val1.x * val2.y + val1.y * val2.x;

    res[ind] = val;
}

// The pcuPrjPad is of size [nview, nuPad]
__global__ void CopyPrjToPad(
    float* pcuPrjPad, const float* pcuPrj, int iv, size_t nu, size_t nuPad, size_t nv, size_t nview
)
{
    int iu = blockIdx.x * blockDim.x + threadIdx.x;
    int iview = blockIdx.y * blockDim.y + threadIdx.y;

    if (iu >= nu || iview >= nview)
    {
        return;
    }

    pcuPrjPad[iview * nuPad + iu] = pcuPrj[iview * nu * nv + iv * nu + iu];
}

__global__ void CopyPadToPrj(
    const float* pcuPrjPad, float* pcuPrj, int iv, size_t nu, size_t nuPad, size_t nv, size_t nview
)
{
    int iu = blockIdx.x * blockDim.x + threadIdx.x;
    int iview = blockIdx.y * blockDim.y + threadIdx.y;

    if (iu >= nu || iview >= nview)
    {
        return;
    }

    pcuPrj[iview * nu * nv + iv * nu + iu] = pcuPrjPad[iview * nuPad + iu];
}


void GetRamp(
    cufftComplex* pcuFreqKernel,
    size_t nu,
    size_t nview,
    float da,
    int filterType,
    cudaStream_t& stream,
    bool isEqualSpace
)
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
                pWindow[i].x = 0.54f + 0.46f * cosf(2 * PI * i / (float)filterLen);
                pWindow[i].y = 0;
            }
            break;
        case FILTER_HANN:
            for (int i = 0; i < filterLen; i++)
            {
                pWindow[i].x = 0.5f + 0.5f * cosf(2 * PI * i / (float)filterLen);
                pWindow[i].y = 0;
            }
            break;
        case FILTER_COSINE:
            for (int i = 0; i < filterLen; i++)
            {
                pWindow[i].x = abs(cosf(PI * i / (float)filterLen));
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