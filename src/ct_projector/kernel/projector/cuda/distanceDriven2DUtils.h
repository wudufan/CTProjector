#pragma once
#include "projector.h"

#include <cuda_runtime.h>

__global__ void PreweightBPKernel(
    float* pPrjs, const float* pDeg, size_t nview, const Detector det, float dx, float dy
);
__global__ void PreweightBPParallelKernel(
	float* pPrjs, const float* pDeg, size_t nview, const Detector det, float dx, float dy
);
__global__ void ReverseWeightBPParallelKernel(
	float* pPrjs, const float* pDeg, size_t nview, const Detector det, float dx, float dy
);

__device__ float GetProjectionOnDetector(float x, float y, float dsd, float dso, float cosDeg, float sinDeg);
__device__ float GetProjectionOnDetectorParallel(float x, float y, float cosDeg, float sinDeg);
__device__ float GetFBPWeight(float x, float y, float dsd, float dso, float cosDeg, float sinDeg);