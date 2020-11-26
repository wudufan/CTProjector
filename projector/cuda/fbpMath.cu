#include "fbpMath.h"

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

__global__ void FilterByFreqMultiply1D(cufftComplex* res, const cufftComplex* src, const cufftComplex* filter, size_t nx, size_t ny)
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