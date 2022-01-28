#include "cudaMath.h"

// res = op1 - lambda * op2
__global__ void Minus2D(
	float* res,
	const float* op1,
	const float* op2,
	int nx,
	int ny,
	int strideRes,
	int stride1,
	int stride2,
	float lambda
)
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
__global__ void Multiply2D(
	float* res,
	const float* op1,
	const float* op2,
	int nx,
	int ny,
	int strideRes,
	int stride1,
	int stride2
)
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
__global__ void Divide2D(
	float* res,
	const float* op1,
	const float* op2,
	int nx,
	int ny,
	int strideRes,
	int stride1,
	int stride2,
	float epsilon
)
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
__global__ void Scale2D(
	float* res,
	const float* op,
	float scale,
	int nx,
	int ny,
	int strideRes,
	int strideOp
)
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
__global__ void ValueCrop2D(
	float* res,
	const float* op,
	float lb,
	float ub,
	int nx,
	int ny,
	int strideRes,
	int strideOp
)
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


__device__ float ClampFloat(float x, float start, float end)
{
	if (x < start)
	{
		return start;
	}
	else if (x >= end)
	{
		return end - 1;
	}
	else
	{
		return x;
	}
}


// clamp x to range [start, end)
__device__ int Clamp(int x, int start, int end)
{
	if (x < start)
	{
		return start;
	}
	else if (x >= end)
	{
		return end - 1;
	}
	else
	{
		return x;
	}
}

// 2d interpolation
__device__ double InterpolateXY(const double* buff, float x, float y, int iz, size_t nx, size_t ny, size_t nz)
{
	x = ClampFloat(x, 0, nx);
	y = ClampFloat(y, 0, ny);

	int ix = int(x);
	int iy = int(y);
	int ix1 = ix + 1;
	int iy1 = iy + 1;

//	ix = Clamp(ix, 0, nx);
//	iy = Clamp(iy, 0, ny);
	ix1 = Clamp(ix1, 0, nx);
	iy1 = Clamp(iy1, 0, ny);

	double wx = (x - ix);
	double wy = (y - iy);

	return double(
		buff[iz * nx * ny + iy * nx + ix] * (1 - wx) * (1 - wy)
		+ buff[iz * nx * ny + iy * nx + ix1] * wx * (1 - wy) 
		+ buff[iz * nx * ny + iy1 * nx + ix] * (1 - wx) * wy 
		+ buff[iz * nx * ny + iy1 * nx + ix1] * wx * wy
	);

}

// 2d interpolation
__device__ float InterpolateXY(const float* buff, float x, float y, int iz, size_t nx, size_t ny, size_t nz, bool truncate)
{
	if (truncate)
	{
		if (x < 0 || x >= nx || y < 0 || y >= ny)
		{
			return 0;
		}
	}

	x = ClampFloat(x, 0, nx);
	y = ClampFloat(y, 0, ny);

	int ix = int(x);
	int iy = int(y);
	int ix1 = ix + 1;
	int iy1 = iy + 1;

//	ix = Clamp(ix, 0, nx);
//	iy = Clamp(iy, 0, ny);
	ix1 = Clamp(ix1, 0, nx);
	iy1 = Clamp(iy1, 0, ny);

	float wx = (x - ix);
	float wy = (y - iy);

	return float(
		buff[iz * nx * ny + iy * nx + ix] * (1 - wx) * (1 - wy)
		+ buff[iz * nx * ny + iy * nx + ix1] * wx * (1 - wy)
		+ buff[iz * nx * ny + iy1 * nx + ix] * (1 - wx) * wy
		+ buff[iz * nx * ny + iy1 * nx + ix1] * wx * wy
	);

}

// 2d interpolation
__device__ float InterpolateXZ(const float* buff, float x, int iy, float z, size_t nx, size_t ny, size_t nz, bool truncate)
{
	if (truncate)
	{
		if (x < 0 || x >= nx || z < 0 || z >= nz)
		{
			return 0;
		}
	}

	x = ClampFloat(x, 0, nx);
	z = ClampFloat(z, 0, nz);

	int ix = int(x);
	int iz = int(z);
	int ix1 = ix + 1;
	int iz1 = iz + 1;

//	ix = Clamp(ix, 0, nx);
//	iy = Clamp(iy, 0, ny);
	ix1 = Clamp(ix1, 0, nx);
	iz1 = Clamp(iz1, 0, nz);

	float wx = (x - ix);
	float wz = (z - iz);

	return float(
		buff[iz * nx * ny + iy * nx + ix] * (1 - wx) * (1 - wz)
		+ buff[iz * nx * ny + iy * nx + ix1] * wx * (1 - wz)
		+ buff[iz1 * nx * ny + iy * nx + ix] * (1 - wx) * wz
		+ buff[iz1 * nx * ny + iy * nx + ix1] * wx * wz
	);

}

// 2d interpolation
__device__ float InterpolateYZ(const float* buff, int ix, float y, float z, size_t nx, size_t ny, size_t nz, bool truncate)
{
	if (truncate)
	{
		if (y < 0 || y >= ny || z < 0 || z >= nz)
		{
			return 0;
		}
	}

	y = ClampFloat(y, 0, ny);
	z = ClampFloat(z, 0, nz);

	int iy = int(y);
	int iz = int(z);
	int iy1 = iy + 1;
	int iz1 = iz + 1;

//	ix = Clamp(ix, 0, nx);
//	iy = Clamp(iy, 0, ny);
	iy1 = Clamp(iy1, 0, ny);
	iz1 = Clamp(iz1, 0, nz);

	float wy = (y - iy);
	float wz = (z - iz);

	return float(
		buff[iz * nx * ny + iy * nx + ix] * (1 - wy) * (1 - wz)
		+ buff[iz * nx * ny + iy1 * nx + ix] * wy * (1 - wz)
		+ buff[iz1 * nx * ny + iy * nx + ix] * (1 - wy) * wz
		+ buff[iz1 * nx * ny + iy1 * nx + ix] * wy * wz
	);

}
