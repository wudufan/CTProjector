#pragma once
#include "projector.h"

#include <cuda_runtime.h>

// struct for siddon tracing
struct SiddonTracingVars
{
	float3 alpha;			// the position of next intersection
	float3 dAlpha;			// the step size between intersections
	int3 ng;				// the grid position
	int3 isPositive;		// if the tracing direction is positive (=1) or not (=0)
	float alphaNow;			// the current tracing position
	float alphaPrev;		// the previous tracing position
	float rayLength;		// total length of the ray
};

__device__ float SiddonRayTracing(float* pPrj, const float* pImg, float3 src, float3 dst, const Grid grid);

__device__ float SiddonRayTracingTransposeAtomicAdd(float* pImg, float val, float3 src, float3 dst, const Grid grid);

__device__ __host__ void MoveSourceDstNearGrid(float3& src, float3&dst, const Grid grid);


