#include "cudaMath.h"

// res = op1 - lambda * op2
__global__ void Minus2D(float* res, const float* op1, const float* op2,
		int nx, int ny, int strideRes, int stride1, int stride2, float lambda)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix >= nx || iy >= ny)
	{
		return;
	}

	res[iy * strideRes + ix] = op1[iy * stride1 + ix]  - lambda * op2[iy * stride2 + ix];
}

// res = op1 * op2
__global__ void Multiply2D(float* res, const float* op1, const float* op2,
		int nx, int ny, int strideRes, int stride1, int stride2)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix >= nx || iy >= ny)
	{
		return;
	}

	res[iy * strideRes + ix] = op1[iy * stride1 + ix] * op2[iy * stride2 + ix];
}

// res = op1 / (op2 + epsilon)
__global__ void Divide2D(float* res, const float* op1, const float* op2,
		int nx, int ny, int strideRes, int stride1, int stride2, float epsilon)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix >= nx || iy >= ny)
	{
		return;
	}

	res[iy * strideRes + ix] = op1[iy * stride1 + ix] / (op2[iy * stride2 + ix] + epsilon);
}


// res = op * scale
__global__ void Scale2D(float* res, const float* op, float scale,
		int nx, int ny, int strideRes, int strideOp)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix >= nx || iy >= ny)
	{
		return;
	}

	res[iy * strideRes + ix] = op[iy * strideOp + ix] * scale;
}

// res[res < lb] = lb
// res[res > ub] = ub
__global__ void ValueCrop2D(float* res, const float* op, float lb, float ub,
		int nx, int ny, int strideRes, int strideOp)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix >= nx || iy >= ny)
	{
		return;
	}

	float val = op[iy * strideOp + ix];
	if (val < lb)
	{
		res[iy * strideRes + ix] = lb;
	}
	else if (val > ub)
	{
		res[iy * strideRes + ix] = ub;
	}
	else
	{
		res[iy * strideRes + ix] = val;
	}
}

void GetThreadsForXZ(dim3 &threads, dim3 &blocks, int nx, int ny, int nz)
{
	threads = dim3(32,1,16);
	if (ny < 8 && nz < 8)
	{
		threads = dim3(512, 1, 1);
	}
	else if (ny >= 8 && nz < 8)
	{
		threads = dim3(32, 16, 1);
	}

	blocks = dim3(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), ceilf(nz / (float)threads.z));
}

void GetThreadsForXY(dim3 &threads, dim3 &blocks, int nx, int ny, int nz)
{
	threads = dim3(32,16,1);
	if (ny < 8 && nz < 8)
	{
		threads = dim3(512, 1, 1);
	}
	else if (nz >= 8 && ny < 8)
	{
		threads = dim3(32, 1, 16);
	}

	blocks = dim3(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), ceilf(nz / (float)threads.z));
}
