/*
GPU code for total variation prior.
*/

#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

/*
3D Aisotropic TV. This is the kernel to calculate the variation image.
Using same padding at the edge.

var - output: TV value at each voxel.
img - input image.
wx, wy, wz - weights for each direction.
nx, ny, nz - size of the image
eps - smoothness at the origin
*/
__global__ void Variation3DKernel(
	float* var, const float* img, float wx, float wy, float wz, size_t nx, size_t ny, size_t nz, float eps
)
{
	size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
	size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
	size_t iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= nx || iy >= ny || iz >= nz)
	{
		return;
	}

	size_t ind = iz * nx * ny + iy * nx + ix;
	float x = img[ind];

	float dx = 0;
	float dy = 0;
	float dz = 0;
	if (ix > 0)
	{
		dx = x - img[ind - 1];
	}

	if (iy > 0)
	{
		dy = x - img[ind - nx];
	}

	if (iz > 0)
	{
		dz = x - img[ind - nx * ny];
	}

	var[ind] = sqrtf(wx * dx * dx + wy * dy * dy + wz * dz * dz + eps);
}

/*
3D Aisotropic TV. This is the kernel for SQS variables.
Using same padding at the edge.

s1 - output: first order derivative in the surrogate (numerator).
s2 - output: second order derivative in the surrogate (denominator).
var - TV value at each voxel.
img - input image.
wx, wy, wz - weights for each direction.
nx, ny, nz - size of the image
*/
__global__ void TVSQS3DKernel(
	float* s1, float* s2, const float* var, const float* img, float wx, float wy, float wz, size_t nx, size_t ny, size_t nz
)
{
	size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
	size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
	size_t iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= nx || iy >= ny || iz >= nz)
	{
		return;
	}

	size_t ind = iz * nx * ny + iy * nx + ix;
	float x = img[ind];
	float v = var[ind];

	float dx0 = 0;
	float dx1 = 0;
	float dy0 = 0;
	float dy1 = 0;
	float dz0 = 0;
	float dz1 = 0;
	float varx, vary, varz;

	if (ix > 0)
	{
		dx0 = x - img[ind - 1];
	}
	if (ix < nx - 1)
	{
		dx1 = x - img[ind + 1];
		varx = var[ind + 1];
	}
	else
	{
		varx = v;  	// approximate var(nx,j) with var(nx-1,j)
	}

	if (iy > 0)
	{
		dy0 = x - img[ind - nx];
	}
	if (iy < ny - 1)
	{
		dy1 = x - img[ind + nx];
		vary = var[ind + nx];
	}
	else
	{
		vary = v;	// approximate var(i, ny) with var(i, ny-1)
	}

	if (iz > 0)
	{
		dz0 = x - img[ind - nx * ny];
	}
	if (iz < nz - 1)
	{
		dz1 = x - img[ind + nx * ny];
		varz = var[ind + nx * ny];
	}
	else
	{
		varz = v;
	}

    dx0 *= wx;
    dx1 *= wx;
    dy0 *= wy;
    dy1 *= wy;
    dz0 *= wz;
    dz1 *= wz;

	s1[ind] = (dx0 + dy0 + dz0) / v + dx1 / varx + dy1 / vary + dz1 / varz;
	s2[ind] = (wx + wy + wz) / v + wx / varx + wy / vary + wz / varz;

}

/*
s1 - first order derivative of the TV prior, to be calculated
s2 - second order derivative of the TV prior, to be calculated
var - the variation at each pixel, to be calculated
img - the original image
wx, wy, wz - the weights along x, y, and z
nBatches, nx, ny, nz - image dimension. s1, s2, var, img are by the shape (nBatches, nz, ny, nx)
eps - the smoothing factor at x=0 for TV
*/
void TVSQS3D_gpu(
	float* s1,
	float* s2,
	float* var,
	const float* img,
	float wx,
	float wy,
	float wz,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	float eps
)
{
	dim3 threads(32, 16, 1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), nz);

	for (int i = 0; i < nBatches; i++) {
		int offset = i * nx * ny * nz;

		Variation3DKernel<<<blocks, threads>>>(var + offset, img + offset, wx, wy, wz, nx, ny, nz, eps);
    	TVSQS3DKernel<<<blocks, threads>>>(s1 + offset, s2 + offset, var + offset, img + offset, wx, wy, wz, nx, ny, nz);
	}

}


extern "C" int cupyTVSQS3D(
	float* s1,
	float* s2,
	float* var,
	const float* img,
	float wx,
	float wy,
	float wz,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	float eps = 1e-8f
) {
	try
	{
		TVSQS3D_gpu(s1, s2, var, img, wx, wy, wz, nBatches, nx, ny, nz, eps);
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cupyTVSQS3D failed: " << e.what() << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();
}

extern "C" int cTVSQS3D(
	float* s1,
	float* s2,
	float* var,
	const float* img,
	float wx,
	float wy,
	float wz,
	size_t nBatches,
	size_t nx,
	size_t ny,
	size_t nz,
	float eps = 1e-8f
)
{
	float* pcus1 = NULL;
	float* pcus2 = NULL;
	float* pcuVar = NULL;
	float* pcuImg = NULL;

	try
	{
		size_t N = nx * ny * nz;

		if (cudaSuccess != cudaMalloc(&pcus1, sizeof(float) * N))
		{
			throw runtime_error("pcus1 allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcus2, sizeof(float) * N))
		{
			throw runtime_error("pcus2 allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuVar, sizeof(float) * N))
		{
			throw runtime_error("pcuVar allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * N))
		{
			throw runtime_error("pcuImg allocation failed");
		}

		cudaMemcpy(pcuImg, img, sizeof(float) * N, cudaMemcpyHostToDevice);

		TVSQS3D_gpu(pcus1, pcus2, pcuVar, pcuImg, wx, wy, wz, nBatches, nx, ny, nz, eps);

		cudaMemcpy(s1, pcus1, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(s2, pcus2, sizeof(float) * N, cudaMemcpyDeviceToHost);
		cudaMemcpy(var, pcuVar, sizeof(float) * N, cudaMemcpyDeviceToHost);
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cTVSQS3D failed: " << e.what() << "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	if (pcus1 != NULL) cudaFree(pcus1);
	if (pcus2 != NULL) cudaFree(pcus2);
	if (pcuVar != NULL) cudaFree(pcuVar);
	if (pcuImg != NULL) cudaFree(pcuImg);

	return cudaGetLastError();

}