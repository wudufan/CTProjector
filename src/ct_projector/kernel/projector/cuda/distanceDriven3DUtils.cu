/*
Utility functions for distance driven 3d
*/

#include "distanceDriven3DUtils.h"
#include "cudaMath.h"


// preweight the projection data by the approximated ray length in a pixel
__global__ void PreweightBPCartKernelXY(
	float* pPrjs,
	const int* iviews,
	size_t nValidViews,
	size_t nview,
	const float3* pDetCenter,
	const float3* pSrc,
	float dz,
	const Detector det
)
{
	int iu = blockIdx.x * blockDim.x + threadIdx.x;
	int iv = blockIdx.y * blockDim.y + threadIdx.y;
	int iviewInd = blockIdx.z * blockDim.z + threadIdx.z;

	if (iu >= det.nu || iviewInd >= nValidViews || iv >= det.nv)
	{
		return;
	}

	int iview = iviews[iviewInd];

	float u = (iu - det.off_u - (det.nu - 1) / 2.0f) * det.du;
	float v = (iv - det.off_v - (det.nv - 1) / 2.0f) * det.dv;
	float3 dst = UVToCart(u, v, pDetCenter[iview], make_float3(1,0,0), make_float3(0,1,0));
	float3 src = pSrc[iview];

	pPrjs[iview * det.nu * det.nv + iv * det.nu + iu] *= dz / fabsf((src.z - dst.z)) * sqrtf(
			(src.z - dst.z) * (src.z - dst.z) + (src.y - dst.y) * (src.y - dst.y) + (src.x - dst.x) * (src.x - dst.x)
		);
}

// for fast calculation, assumed detU = (1,0,0) and detV = (0,1,0)
// directly convert to image coordinate
// Cart means that the detector aligns with the cartesian coordinate
__device__ float2 ProjectConeToDetCart(float3 pt, float3 detCenter, float3 src, const Detector& det)
{
	float r = (detCenter.z - src.z) / (pt.z - src.z);
	float u = src.x - detCenter.x + r * (pt.x - src.x);
	float v = src.y - detCenter.y + r * (pt.y - src.y);

	u = u / det.du + det.off_u + (det.nu - 1) / 2.0f;
	v = v / det.dv + det.off_v + (det.nv - 1) / 2.0f;

	return make_float2(u, v);

}
