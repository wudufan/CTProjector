#pragma once

// res = op1 - lambda * op2
__global__ void Minus2D(float* res, const float* op1, const float* op2,
		int nx, int ny, int strideRes, int stride1, int stride2, float lambda = 1);

// res = op1 * op2
__global__ void Multiply2D(float* res, const float* op1, const float* op2,
		int nx, int ny, int strideRes, int stride1, int stride2);

// res = op1 / (op2 + epsilon)
__global__ void Divide2D(float* res, const float* op1, const float* op2,
		int nx, int ny, int strideRes, int stride1, int stride2, float epsilon = 0);

// res = op * scale
__global__ void Scale2D(float* res, const float* op, float scale,
		int nx, int ny, int strideRes, int strideOp);

// res[res < lb] = lb
// res[res > ub] = ub
__global__ void ValueCrop2D(float* res, const float* op, float lb, float ub,
		int nx, int ny, int strideRes, int strideOp);

// v1 - v2
inline __device__ __host__ float3 vecMinus(float3 v1, float3 v2)
{
	return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

// v1 dot v2
inline __device__ __host__ float vecDot(float3 v1, float3 v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

// len(v)^2
inline __device__ __host__ float vecLength2(float3 v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

// len(v)
inline __device__ __host__ float vecLength(float3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// v / len(v)
inline __device__ __host__ float3 vecNormalize(float3 v)
{
	float len = vecLength(v);
	if (len < 1e-6f)
	{
		return v;
	}
	else
	{
		return make_float3(v.x / len, v.y / len, v.z / len);
	}
}

void GetThreadsForXZ(dim3 &threads, dim3 &blocks, int nx, int ny, int nz);

void GetThreadsForXY(dim3 &threads, dim3 &blocks, int nx, int ny, int nz);
