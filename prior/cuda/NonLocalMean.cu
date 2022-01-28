/*
GPU code for non-local guided-filtering (HYPR_NLM) and guided non-local mean (NLM)
For ordinary NLM: use the same guide image as the input image;
For Gaussian denoising: use a guide image filled with 1. Set the gaussian kernel size to 1.
*/

#include <math.h>
#include <iostream>
#include <stdexcept>
#include <sstream>

using namespace std;

#define MIN(x,y) ((x) < (y) ? (x) : (y))
#define MAX(x,y) ((x) > (y) ? (x) : (y))

__device__ int clamp(int ix, int nx)
{
    return MIN(MAX(ix, 0), nx-1);
}

/*
Calculate the distance between two points in guide using gaussian-weighted non-local distance

guide - guide image;
gaussian - the gaussian kernel;
ix0,iy0,iz0 - point1 in the guide image;
ix1,iy1,iz1 - point2 in the guide image;
nx,ny,nz - size of the guide image;
nkx, nky, nkz - size of the guassian kernel
*/
__device__ float GaussianDistance2(
	const float* guide,
	const float* gaussian,
    int ix0,
	int iy0,
	int iz0,
	int ix1,
	int iy1,
	int iz1,
    size_t nx,
	size_t ny,
	size_t nz,
	int nkx,
	int nky,
	int nkz
)
{
	float dist = 0;
	for (int ikx = 0; ikx < nkx; ikx++)
	{
		for (int iky = 0; iky < nkx; iky++)
		{
			for (int ikz = 0; ikz < nkx; ikz++)
			{
                float weight = gaussian[ikz * nkx * nky + iky * nkx + ikx];

                int ix = clamp(ix0 + ikx - nkx / 2, nx);
                int iy = clamp(iy0 + iky - nky / 2, ny);
                int iz = clamp(iz0 + ikz - nkz / 2, nz);
                float val0 = guide[iz * nx * ny + iy * nx + ix];

                ix = clamp(ix1 + ikx - nkx / 2, nx);
                iy = clamp(iy1 + iky - nky / 2, ny);
                iz = clamp(iz1 + ikz - nkz / 2, nz);
                float val1 = guide[iz * nx * ny + iy * nx + ix];

				dist += weight * (val0 - val1) * (val0 - val1);
			}
		}
	}

	return dist;
}

/*
This is the kernel function for HYPR-NLM. It does a non-local denoising for both img and guide. Then for each search window, it calclates the ratio between img and guide. 
At last, it multiply the guide by the ratio to achieve a "local linear fitting"

res - the result buffer
img - the image to be denoised
guide - the guide image
gaussian - the gaussian smoothing window to calculate non-local distance
d2 - the large d2 is, the smaller the guide image's effect will be
eps - a regularization factor when divding the guide image
nsx, nsy, nsz - the local search window to perform local linear fitting
nx, ny, nz - the size of the image
nkx, nky, nkz - the size of the gaussian kernel
*/
__global__ void hyprNlmKernel(
	float* res,
	const float* img,
	const float* guide,
	const float* gaussian,
    float d2,
	float eps,
	int nsx,
	int nsy,
	int nsz,
	size_t nx,
	size_t ny,
	size_t nz,
	int nkx,
	int nky,
	int nkz
)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= nx || iy >= ny || iz >= nz)
	{
		return;
	}

	float sumImg = 0;
	float sumGuide = 0;
	for (int isx = 0; isx < nsx; isx++)
	{
		for (int isy = 0; isy < nsy; isy++)
		{
			for (int isz = 0; isz < nsz; isz++)
			{
                int ix1 = ix - nsx / 2 + isx;
                int iy1 = iy - nsy / 2 + isy;
                int iz1 = iz - nsz / 2 + isz;
                float dist2 = GaussianDistance2(guide, gaussian, ix, iy, iz, ix1, iy1, iz1, nx, ny, nz, nkx, nky, nkz);
                float weight = expf(-dist2 / d2);
                
                ix1 = clamp(ix1, nx);
                iy1 = clamp(iy1, ny);
                iz1 = clamp(iz1, nz);
				sumImg += weight * img[iz1 * nx * ny + iy1 * nx + ix1];
				sumGuide += weight * guide[iz1 * nx * ny + iy1 * nx + ix1];
			}
		}
	}

	res[iz * nx * ny + iy * nx + ix] = sumImg / (sumGuide + eps) * guide[iz * nx * ny + iy * nx + ix];

}

/*
The non-local mean kernel.
It use the guide image to calculate the weighting factors then average the image within the earch window. 
*/
__global__ void nlmKernel(
	float* res,
	const float* img,
	const float* guide,
	const float* gaussian,
    float d2,
	float eps,
	int nsx,
	int nsy,
	int nsz,
	size_t nx,
	size_t ny,
	size_t nz,
	int nkx,
	int nky,
	int nkz
)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= nx || iy >= ny || iz >= nz)
	{
		return;
	}

	float sumImg = 0;
	float sumGuide = 0;
	for (int isx = 0; isx < nsx; isx++)
	{
		for (int isy = 0; isy < nsy; isy++)
		{
			for (int isz = 0; isz < nsz; isz++)
			{
                int ix1 = ix - nsx / 2 + isx;
                int iy1 = iy - nsy / 2 + isy;
                int iz1 = iz - nsz / 2 + isz;
                float dist2 = GaussianDistance2(guide, gaussian, ix, iy, iz, ix1, iy1, iz1, nx, ny, nz, nkx, nky, nkz);
                float weight = expf(-dist2 / d2);
                
                ix1 = clamp(ix1, nx);
                iy1 = clamp(iy1, ny);
                iz1 = clamp(iz1, nz);
				sumImg += weight * img[iz1 * nx * ny + iy1 * nx + ix1];
				sumGuide += weight;
			}
		}
	}

	res[iz * nx * ny + iy * nx + ix] = sumImg / (sumGuide + eps);

}

extern "C" int cSetDevice(int i)
{
	return cudaSetDevice(i);
}

void hyprNlm(
	float* res,
	const float* img,
	const float* guide,
	const float* gaussian,
    float d2,
	float eps,
	int nsx,
	int nsy,
	int nsz,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	int nkx,
	int nky,
	int nkz
)
{
    dim3 threads(32, 16, 1);
    dim3 blocks((int)ceilf(nx / (float)threads.x), (int)ceilf(ny / (float)threads.y), (int)ceilf(nz / (float)threads.z));
    for (int ib = 0; ib < nBatches; ib++)
    {
        int offset = ib * nx * ny * nz;
        hyprNlmKernel<<<blocks, threads>>>(
			res + offset,
			img + offset,
			guide + offset, 
            gaussian,
			d2,
			eps,
			nsx,
			nsy,
			nsz,
			nx,
			ny,
			nz,
			nkx,
			nky,
			nkz
		);
    }
    cudaDeviceSynchronize();

}

void nlm(
	float* res,
	const float* img,
	const float* guide,
	const float* gaussian,
    float d2,
	float eps,
	int nsx,
	int nsy,
	int nsz,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	int nkx,
	int nky,
	int nkz
)
{
    dim3 threads(32, 16, 1);
    dim3 blocks((int)ceilf(nx / (float)threads.x), (int)ceilf(ny / (float)threads.y), (int)ceilf(nz / (float)threads.z));
    for (int ib = 0; ib < nBatches; ib++)
    {
        int offset = ib * nx * ny * nz;
        nlmKernel<<<blocks, threads>>>(
			res + offset,
			img + offset,
			guide + offset, 
            gaussian,
			d2,
			eps,
			nsx,
			nsy,
			nsz,
			nx,
			ny,
			nz,
			nkx,
			nky,
			nkz
		);
    }
    cudaDeviceSynchronize();

}

extern "C" int cHyprNlm(
	float* res,
	const float* img,
	const float* guide,
	const float* gaussian,
    float d2,
	float eps,
	int nsx,
	int nsy,
	int nsz,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	int nkx,
	int nky,
	int nkz
)
{
	float* cuRes = NULL;
	float* cuGaussian = NULL;
	float* cuImg = NULL;
	float* cuGuide = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&cuRes, sizeof(float) * nx * ny * nz * nBatches))
		{
			throw runtime_error("cuRes allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&cuGaussian, sizeof(float) * nkx * nky * nkz))
		{
			throw runtime_error("cuGaussian allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&cuImg, sizeof(float) * nx * ny * nz * nBatches))
		{
			throw runtime_error("cuImg allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&cuGuide, sizeof(float) * nx * ny * nz * nBatches))
		{
			throw runtime_error("cuGuide allocation failed");
        }

        cudaMemcpy(cuImg, img, sizeof(float) * nx * ny * nz * nBatches, cudaMemcpyHostToDevice);
        cudaMemcpy(cuGuide, guide, sizeof(float) * nx * ny * nz * nBatches, cudaMemcpyHostToDevice);
        cudaMemcpy(cuGaussian, gaussian, sizeof(float) * nkx * nky * nkz, cudaMemcpyHostToDevice);
        
        hyprNlm(cuRes, cuImg, cuGuide, cuGaussian, d2, eps, nsx, nsy, nsz, nBatches, nx, ny, nz, nkx, nky, nkz);

		cudaMemcpy(res, cuRes, sizeof(float) * nx * ny * nz * nBatches, cudaMemcpyDeviceToHost);
	}
	catch (std::exception &e)
	{
        ostringstream oss;
		oss << "cHyprNlm() failed: " << e.what() << " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	if (cuRes != NULL)
    {
        cudaFree(cuRes);
    }
    if (cuGaussian != NULL)
    {
        cudaFree(cuGaussian);
    }
    if (cuImg != NULL)
    {
        cudaFree(cuImg);
    }
    if (cuGuide != NULL)
    {
        cudaFree(cuGuide);
    }

	return cudaGetLastError();

}

extern "C" int cNlm(
	float* res,
	const float* img,
	const float* guide,
	const float* gaussian,
    float d2,
	float eps,
	int nsx,
	int nsy,
	int nsz,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	int nkx,
	int nky,
	int nkz
)
{
	float* cuRes = NULL;
	float* cuGaussian = NULL;
	float* cuImg = NULL;
	float* cuGuide = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&cuRes, sizeof(float) * nx * ny * nz * nBatches))
		{
			throw runtime_error("cuRes allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&cuGaussian, sizeof(float) * nkx * nky * nkz))
		{
			throw runtime_error("cuGaussian allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&cuImg, sizeof(float) * nx * ny * nz * nBatches))
		{
			throw runtime_error("cuImg allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&cuGuide, sizeof(float) * nx * ny * nz * nBatches))
		{
			throw runtime_error("cuGuide allocation failed");
        }
        
        cudaMemcpy(cuImg, img, sizeof(float) * nx * ny * nz * nBatches, cudaMemcpyHostToDevice);
        cudaMemcpy(cuGuide, guide, sizeof(float) * nx * ny * nz * nBatches, cudaMemcpyHostToDevice);
        cudaMemcpy(cuGaussian, gaussian, sizeof(float) * nkx * nky * nkz, cudaMemcpyHostToDevice);

        nlm(cuRes, cuImg, cuGuide, cuGaussian, d2, eps, nsx, nsy, nsz, nBatches, nx, ny, nz, nkx, nky, nkz);

		cudaMemcpy(res, cuRes, sizeof(float) * nx * ny * nz * nBatches, cudaMemcpyDeviceToHost);
	}
	catch (std::exception &e)
	{
        ostringstream oss;
		oss << "cNlm() failed: " << e.what() << " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	if (cuRes != NULL)
    {
        cudaFree(cuRes);
    }
    if (cuGaussian != NULL)
    {
        cudaFree(cuGaussian);
    }
    if (cuImg != NULL)
    {
        cudaFree(cuImg);
    }
    if (cuGuide != NULL)
    {
        cudaFree(cuGuide);
    }

	return cudaGetLastError();

}

extern "C" int cupyHyprNlm(
	float* res,
	const float* img,
	const float* guide,
	const float* gaussian,
    float d2,
	float eps,
	int nsx,
	int nsy,
	int nsz,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	int nkx,
	int nky,
	int nkz
)
{
	try
	{
        hyprNlm(res, img, guide, gaussian, d2, eps, nsx, nsy, nsz, nBatches, nx, ny, nz, nkx, nky, nkz);
	}
	catch (std::exception &e)
	{
        ostringstream oss;
		oss << "cupyHyprNlm() failed: " << e.what() << " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();

}

extern "C" int cupyNlm(
	float* res,
	const float* img,
	const float* guide,
	const float* gaussian,
    float d2,
	float eps,
	int nsx,
	int nsy,
	int nsz,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	int nkx,
	int nky,
	int nkz
)
{
	try
	{
        nlm(res, img, guide, gaussian, d2, eps, nsx, nsy, nsz, nBatches, nx, ny, nz, nkx, nky, nkz);
	}
	catch (std::exception &e)
	{
        ostringstream oss;
		oss << "cupyNlm() failed: " << e.what() << " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();

}
