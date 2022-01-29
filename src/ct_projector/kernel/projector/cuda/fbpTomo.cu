#include "fbp.h"
#include "cudaMath.h"
#include "siddon.h"
#include "fbpMath.h"

#include <math.h>
#include <cufft.h>
#include <stdexcept>
#include <sstream>
#include <iostream>

using namespace std;

void fbpTomo::Setup(
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
    float du,
    float dv,
    float off_u,
    float off_v,
    float dsd,
    float dso,
    int typeProjector,
    float cutoffX,
    float cutoffZ
)
{
    Projector::Setup(
        nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
        nu, nv, nview, du, dv, off_u, off_v, dsd, dso, typeProjector
    );

    this->cutoffX = cutoffX;
    this->cutoffZ = cutoffZ;
}

float GetFrequency(int i, int filterLen, float du, float dx, float angleWeight, float cutoff)
{
    float fu, wx;
    if (i > (filterLen + 1) / 2)
    {
        fu = (i - filterLen) / float(filterLen) / du;
    }
    else
    {
        fu = i / float(filterLen) / du;
    }
    wx = fu * angleWeight * dx * 2;
    if (wx > cutoff)
    {
        wx = cutoff;
    }
    
    return wx;
}

// always equispace
void GetRampTomo(
    cufftComplex* pcuFreqKernel,
    float3* pDetCenter,
    float3* pSrc,
    size_t nview,
    size_t nu,
    float du,
    float dx,
    float dz,
    float cutoffX,
    float cutoffZ,
    int filterType,
    cudaStream_t& stream
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
            throw std::runtime_error("cufftPlan1d failure");
        }
    
        if (CUFFT_SUCCESS != cufftSetStream(plan, stream))
        {
            throw std::runtime_error("cudaSetStream failure");
        }
    
        // RL kernel
        if (cudaSuccess != cudaMalloc(&pcuRamp, sizeof(cufftComplex) * filterLen))
        {
            throw std::runtime_error("pcuRamp allocation error");
        }

        if (cudaSuccess != cudaMalloc(&pcuRealKernel, sizeof(cufftReal) * filterLen))
        {
            throw std::runtime_error("pRealKernel allocation error");
        }

        pRealKernel = new cufftReal [filterLen];
        pRamp = new cufftComplex [filterLen];
        pWindow = new cufftComplex [filterLen];

        // generate filter for each angle
        for (int iview = 0; iview < nview; iview++)
        {
            cout << iview << endl;

            // get the angle of the current projection
            // beta is the angle from x axis to the frequency plane
            float3 src = pSrc[iview];
            float3 det = pDetCenter[iview];
            float r = sqrtf((det.x - src.x) * (det.x - src.x) + (det.z - src.z) * (det.z - src.z));
            float sinDeg = (det.x - src.x) / r;
            float cosDeg = (src.z - det.z) / r;

            // a virtual detector at the frequency plane
            float vdu = du * cosDeg;

            // equispace
            for (int i = 0; i < filterLen; i++)
            {
                int k = i - (nu - 1);
                if (k == 0)
                {
                    pRamp[i].x = 1 / (4 * vdu * vdu);
                }
                else if (k % 2 != 0)
                {
                    pRamp[i].x = -1 / (PI * PI * k * k * vdu * vdu);
                }
                else
                {
                    pRamp[i].x = 0;
                }
                pRamp[i].y = 0;
            }

            cudaMemcpyAsync(pcuRamp, pRamp, sizeof(cufftComplex) * filterLen, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            cufftExecC2C(plan, pcuRamp, pcuRamp, CUFFT_FORWARD);
            cudaMemcpyAsync(pRamp, pcuRamp, sizeof(cufftComplex) * filterLen, cudaMemcpyDeviceToHost, stream);

            // weighting window in frequency domain
            switch(filterType)
            {
            case FILTER_HAMMING:
                // Hamming
                for (int i = 0; i < filterLen; i++)
                {
                    // fftshift
                    float wx = GetFrequency(i, filterLen, vdu, dx, cosDeg, cutoffX);
                    float wz = GetFrequency(i, filterLen, vdu, dz, sinDeg, cutoffZ);

                    pWindow[i].x = (0.54f + 0.46f * cosf(PI * wx / cutoffX)) * (0.54f + 0.46f * cosf(PI * wz / cutoffZ));
                    pWindow[i].y = 0;
                }
                break;
            case FILTER_HANN:
                for (int i = 0; i < filterLen; i++)
                {
                    float wx = GetFrequency(i, filterLen, vdu, dx, cosDeg, cutoffX);
                    float wz = GetFrequency(i, filterLen, vdu, dz, sinDeg, cutoffZ);

                    pWindow[i].x = (0.5f + 0.5f * cosf(PI * wx / cutoffX)) * (0.5f + 0.5f * cosf(PI * wz / cutoffZ));
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

            for (int i = 0; i < filterLen; i++)
            {
                pRealKernel[i] = pRamp[i].x / filterLen;
            }
            cudaMemcpyAsync(pcuRealKernel, pRealKernel, sizeof(cufftReal) * filterLen, cudaMemcpyHostToDevice, stream);

            cufftHandle planR2C;
            cufftPlan1d(&planR2C, filterLen, CUFFT_R2C, 1);
            cufftSetStream(planR2C, stream);
            cufftExecR2C(planR2C, pcuRealKernel, pcuFreqKernel + iview * nu);
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
        oss << "GetRampTomo() failed: " << e.what()
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

void fbpTomo::Filter(float* pcuFPrj, const float* pcuPrj, const float* pcuDetCenter, const float* pcuSrc)
{
    // the filter is carried out for each different projection
    int filterLen = nu * 2 - 1;
    float* pcuPrjPad = NULL;
    cufftComplex* pcuFreqPrj = NULL;
    cufftComplex* pcuFilter = NULL;
    cufftHandle plan = 0;
    cufftHandle planInverse = 0;

    // cpu memory
    float3* pDetCenter = NULL;
    float3* pSrc = NULL;

    try
    {
        pDetCenter = new float3 [nview];
        pSrc = new float3 [nview];
        cudaMemcpyAsync(pDetCenter, pcuDetCenter, sizeof(float3) * nview, cudaMemcpyDeviceToHost, m_stream);
        cudaMemcpyAsync(pSrc, pcuSrc, sizeof(float3) * nview, cudaMemcpyDeviceToHost, m_stream);
        cudaStreamSynchronize(m_stream);

        // projection
        if (cudaSuccess != cudaMalloc(&pcuPrjPad, sizeof(float) * filterLen * nv))
        {
            throw std::runtime_error("pcuPrjPad allocation failure");
        }

        // freq projection
        if (cudaSuccess != cudaMalloc(&pcuFreqPrj, sizeof(cufftComplex) * nu * nv))
        {
            throw std::runtime_error("pcuFreqPrj allocation failure");
        }

        // filter
        if (cudaSuccess != cudaMalloc(&pcuFilter, sizeof(cufftComplex) * nu * nview))
        {
            throw std::runtime_error("pcuFilter allocation failure");
        }
        GetRampTomo(pcuFilter, pDetCenter, pSrc, nview, nu, du, dx, dz, cutoffX, cutoffZ, typeProjector, m_stream);

        // fft plan
        if (CUFFT_SUCCESS != cufftPlanMany(&plan, 1, &filterLen, NULL, 1, filterLen, NULL, 1, nu, CUFFT_R2C, nv))
        {
            throw std::runtime_error("fft plan error");
        }
        cufftSetStream(plan, m_stream);
        
        if (CUFFT_SUCCESS != cufftPlanMany(&planInverse, 1, &filterLen, NULL, 1, nu, NULL, 1, filterLen, CUFFT_C2R, nv))
        {
            throw std::runtime_error("ifft plan error");
        }
        cufftSetStream(plan, m_stream);

        // kernel threads and blocks
        dim3 threads(32, 16, 1);
        dim3 blocks(ceilf(nu / (float)threads.x), ceilf(nv / (float)threads.y), 1);

        for (int ib = 0; ib < nBatches; ib++)
        {
            for (int iview = 0; iview < nview; iview++)
            {
                // get the angle information
                float3 src = pSrc[iview];
                float3 det = pDetCenter[iview];
                float r = sqrtf((det.x - src.x) * (det.x - src.x) + (det.z - src.z) * (det.z - src.z));
                float cosDeg = (src.z - det.z) / r;
                // a virtual detector at the frequency plane
                float vdu = du * cosDeg;

                float scale = PI / nview * vdu / filterLen;

                // projection padding
                cudaMemsetAsync(pcuPrjPad, 0, sizeof(float) * filterLen * nv, m_stream);
                cudaMemcpy2DAsync(
                    pcuPrjPad,
                    filterLen * sizeof(float), 
                    pcuPrj + ib * nview * nu * nv + iview * nu * nv, nu * sizeof(float), 
                    nu * sizeof(float),
                    nv,
                    cudaMemcpyDeviceToDevice,
                    m_stream
                );
                cudaStreamSynchronize(m_stream);
                
                // filter
                cufftExecR2C(plan, pcuPrjPad, pcuFreqPrj);
                FilterByFreqMultiply1D<<<blocks, threads, 0, m_stream>>>(pcuFreqPrj, pcuFreqPrj, pcuFilter, nu, nv);
                cudaStreamSynchronize(m_stream);
                cufftExecC2R(planInverse, pcuFreqPrj, pcuPrjPad);

                // post scaling
                Scale2D<<<blocks, threads, 0, m_stream>>>(
                    pcuPrjPad + nu - 1, pcuPrjPad + nu - 1, scale, nu, nv, filterLen, filterLen
                );

                // get filtered projection
                cudaMemcpy2DAsync(
                    pcuFPrj + ib * nview * nu * nv + iview * nu * nv,
                    nu * sizeof(float), 
                    pcuPrjPad + nu - 1,
                    filterLen * sizeof(float),
                    nu * sizeof(float),
                    nv,
                    cudaMemcpyDeviceToDevice,
                    m_stream
                );
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
        if (pDetCenter != NULL) delete [] pDetCenter;
        if (pSrc != NULL) delete [] pSrc;

        ostringstream oss;
        oss << "fbpTomo::Filter() failed: " << e.what()
            << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
        cerr << oss.str() << endl;
        throw runtime_error(oss.str().c_str());
    }

    if (plan != 0) cufftDestroy(plan);
    if (planInverse != 0) cufftDestroy(planInverse);
    if (pcuPrjPad != NULL) cudaFree(pcuPrjPad);
    if (pcuFreqPrj != NULL) cudaFree(pcuFreqPrj);
    if (pcuFilter != NULL) cudaFree(pcuFilter);
    if (pDetCenter != NULL) delete [] pDetCenter;
    if (pSrc != NULL) delete [] pSrc;

}

extern "C" int cupyFbpTomoFilter(
    float* fprj,
    const float* prj,
    const float* detCenter,
    const float* src,
    int nBatches,
    size_t nu,
    size_t nv,
    size_t nview,
    float du,
    float dx,
    float dz,
    int typeFilter = 0,
    float cutoffX = 1,
    float cutoffZ = 1
)
{
    try
	{
		fbpTomo projector;
		projector.Setup(
            nBatches, 512, 512, 512, dx, 1, dz, 0, 0, 0,
            nu, nv, nview, du, 1, 0, 0, 0, 0, typeFilter, cutoffX, cutoffZ
        );

		projector.Filter(fprj, prj, detCenter, src);

	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cupyFbpTomoFilter() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();

}