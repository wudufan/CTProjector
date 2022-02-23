#pragma once

// provides math for fft

#include <cufft.h>

__global__ void ComplexMultiply2D(cufftComplex* res, const cufftComplex* op1, const cufftComplex* op2, size_t nx, size_t ny);
__global__ void FilterByFreqMultiply1D(cufftComplex* res, const cufftComplex* src, const cufftComplex* filter, size_t nx, size_t ny);

// The pcuPrjPad is of size [nview, nuPad]
__global__ void CopyPrjToPad(float* pcuPrjPad, const float* pcuPrj, int iv, size_t nu, size_t nuPad, size_t nv, size_t nview);
__global__ void CopyPadToPrj(const float* pcuPrjPad, float* pcuPrj, int iv, size_t nu, size_t nuPad, size_t nv, size_t nview);

// get ramp filter
void GetRamp(
    cufftComplex* pcuFreqKernel,
    size_t nu,
    size_t nview,
    float da,
    int filterType,
    cudaStream_t& stream,
    bool isEqualSpace = false
);