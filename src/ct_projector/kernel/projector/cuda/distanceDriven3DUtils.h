/*
Utility functions for distance driven 3d.
*/

#pragma once
#include "projector.h"

#include <cuda_runtime.h>

__global__ void PreweightBPCartKernelXY(
	float* pPrjs,
	const int* iviews,
	size_t nValidViews,
	size_t nview,
	const float3* pDetCenter,
	const float3* pSrc,
	float dz,
	const Detector det
);

__device__ float2 ProjectConeToDetCart(
	float3 pt, float3 detCenter, float3 src, const Detector& det
);