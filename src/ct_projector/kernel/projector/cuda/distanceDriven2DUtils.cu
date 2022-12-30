#include "distanceDriven2DUtils.h"
#include "cudaMath.h"

/*
preweighting for backpojection
It will weight the projections by the esitimated ray length
*/
__global__ void PreweightBPKernel(float* pPrjs, const float* pDeg, size_t nview, const Detector det, float dx, float dy)
{
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iv = blockIdx.y * blockDim.y + threadIdx.y;
	int iview = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iv >= det.nv || iview >= nview)
	{
		return;
	}

	float a = (-(iu - (det.nu-1) / 2.0f) - det.off_u) * det.du;
	float deg = pDeg[iview];

	if (fabsf(__cosf(deg)) > fabsf(__sinf(deg)))
	{
		// Y as main axis
		pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] *= dy / fabsf(__cosf(deg - a));
	}
	else
	{
		//X as main axis
		pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] *= dx / fabsf(__sinf(deg - a));
	}
		
}

/*
preweighting for backpojection
It will weight the projections by the esitimated ray length
*/
__global__ void PreweightBPParallelKernel(
	float* pPrjs, const float* pDeg, size_t nview, const Detector det, float dx, float dy
)
{
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iv = blockIdx.y * blockDim.y + threadIdx.y;
	int iview = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iv >= det.nv || iview >= nview)
	{
		return;
	}

	float deg = pDeg[iview];

	if (fabsf(__cosf(deg)) > fabsf(__sinf(deg)))
	{
		// Y as main axis
		pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] *= dy / fabsf(__cosf(deg));
	}
	else
	{
		//X as main axis
		pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] *= dx / fabsf(__sinf(deg));
	}
		
}

// Cancel the ray-length weighting during forward projection
__global__ void ReverseWeightBPParallelKernel(
	float* pPrjs, const float* pDeg, size_t nview, const Detector det, float dx, float dy
)
{
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iv = blockIdx.y * blockDim.y + threadIdx.y;
	int iview = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iv >= det.nv || iview >= nview)
	{
		return;
	}

	float deg = pDeg[iview];

	if (fabsf(__cosf(deg)) > fabsf(__sinf(deg)))
	{
		// Y as main axis
		pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] /= dy / fabsf(__cosf(deg));
	}
	else
	{
		//X as main axis
		pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] /= dx / fabsf(__sinf(deg));
	}
		
}

__device__ float GetProjectionOnDetector(float x, float y, float dsd, float dso, float cosDeg, float sinDeg)
{
	float rx =  x * cosDeg + y * sinDeg;
	float ry = -x * sinDeg + y * cosDeg;
	return atanf(rx / (ry + dso));

}

__device__ float GetProjectionOnDetectorParallel(float x, float y, float cosDeg, float sinDeg)
{
	return x * cosDeg + y * sinDeg;

}

__device__ float GetFBPWeight(float x, float y, float dsd, float dso, float cosDeg, float sinDeg)
{
	float rx =  x * cosDeg + y * sinDeg;
	float ry = -x * sinDeg + y * cosDeg;

	return dso*dso / (rx*rx + (dso+ry)*(dso+ry));

}