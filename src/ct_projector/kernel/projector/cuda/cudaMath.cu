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

// Clamp x to [start, end]
__device__ float ClampFloat(float x, float start, float end)
{
	if (x < start)
	{
		return start;
	}
	else if (x > end)
	{
		return end;
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

	ix = Clamp(ix, 0, nx);
	iy = Clamp(iy, 0, ny);
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

	ix = Clamp(ix, 0, nx);
	iy = Clamp(iy, 0, ny);
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

	ix = Clamp(ix, 0, nx);
	iy = Clamp(iy, 0, ny);
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

	ix = Clamp(ix, 0, nx);
	iy = Clamp(iy, 0, ny);
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

// Calculate the integral value inside a box
// integral = (pixel sum) - (edge leftout) + (corner leftout)
__device__ float IntegralBoxXY(
	const float* buff, float x1, float y1, float x2, float y2, const int iz, size_t nx, size_t ny, size_t nz
)
{
	x1 = ClampFloat(x1, 0, nx);
	y1 = ClampFloat(y1, 0, ny);
	x2 = ClampFloat(x2, 0, nx);
	y2 = ClampFloat(y2, 0, ny);

	int ix1 = Clamp(int(x1), 0, nx);
	int iy1 = Clamp(int(y1), 0, ny);
	int ix2 = Clamp(int(x2), 0, nx);
	int iy2 = Clamp(int(y2), 0, ny);

	// there is not area to integral on
	if (x2 <= x1 || y2 <= y1)
	{
		return 0;
	}

	float integral = 0;	

	// 1. Calculate the integral of all the pixels
	for (int iy = iy1; iy <= iy2; iy++)
	{
		for (int ix = ix1; ix <= ix2; ix++)
		{
			integral += buff[iz * nx * ny + iy * nx + ix];
		}
	}

	// 2. subtract the integral on the 4 sides
	// left
	for (int iy = iy1; iy <= iy2; iy++)
	{
		integral -= buff[iz * nx * ny + iy * nx + ix1] * (x1 - ix1);
	}
	// right
	for (int iy = iy1; iy <= iy2; iy++)
	{
		integral -= buff[iz * nx * ny + iy * nx + ix2] * (ix2 + 1 - x2);
	}
	// top
	for (int ix = ix1; ix <= ix2; ix++)
	{
		integral -= buff[iz * nx * ny + iy1 * nx + ix] * (y1 - iy1);
	}
	// bottom
	for (int ix = ix1; ix <= ix2; ix++)
	{
		integral -= buff[iz * nx * ny + iy2 * nx + ix] * (iy2 + 1 - y2);
	}

	// 3. add the corner integrals
	integral += buff[iz * nx * ny + iy1 * nx + ix1] * (x1 - ix1) * (y1 - iy1);
	integral += buff[iz * nx * ny + iy1 * nx + ix2] * (ix2 + 1 - x2) * (y1 - iy1);
	integral += buff[iz * nx * ny + iy2 * nx + ix1] * (x1 - ix1) * (iy2 + 1 - y2);
	integral += buff[iz * nx * ny + iy2 * nx + ix2] * (ix2 + 1 - x2) * (iy2 + 1 - y2);

	return integral;

}


// Calculate the integral value inside a box
// integral = (pixel sum) - (edge leftout) + (corner leftout)
__device__ float IntegralBoxXZ(
	const float* buff, float x1, float z1, float x2, float z2, const int iy, size_t nx, size_t ny, size_t nz
)
{
	x1 = ClampFloat(x1, 0, nx);
	z1 = ClampFloat(z1, 0, nz);
	x2 = ClampFloat(x2, 0, nx);
	z2 = ClampFloat(z2, 0, nz);

	int ix1 = Clamp(int(x1), 0, nx);
	int iz1 = Clamp(int(z1), 0, nz);
	int ix2 = Clamp(int(x2), 0, nx);
	int iz2 = Clamp(int(z2), 0, nz);

	// there is not area to integral on
	if (x2 <= x1 || z2 <= z1)
	{
		return 0;
	}

	float integral = 0;	

	// 1. Calculate the integral of all the pixels
	for (int iz = iz1; iz <= iz2; iz++)
	{
		for (int ix = ix1; ix <= ix2; ix++)
		{
			integral += buff[iz * nx * ny + iy * nx + ix];
		}
	}

	// 2. subtract the integral on the 4 sides
	// left
	for (int iz = iz1; iz <= iz2; iz++)
	{
		integral -= buff[iz * nx * ny + iy * nx + ix1] * (x1 - ix1);
	}
	// right
	for (int iz = iz1; iz <= iz2; iz++)
	{
		integral -= buff[iz * nx * ny + iy * nx + ix2] * (ix2 + 1 - x2);
	}
	// top
	for (int ix = ix1; ix <= ix2; ix++)
	{
		integral -= buff[iz1 * nx * ny + iy * nx + ix] * (z1 - iz1);
	}
	// bottom
	for (int ix = ix1; ix <= ix2; ix++)
	{
		integral -= buff[iz2 * nx * ny + iy * nx + ix] * (iz2 + 1 - z2);
	}

	// 3. add the corner integrals
	integral += buff[iz1 * nx * ny + iy * nx + ix1] * (x1 - ix1) * (z1 - iz1);
	integral += buff[iz1 * nx * ny + iy * nx + ix2] * (ix2 + 1 - x2) * (z1 - iz1);
	integral += buff[iz2 * nx * ny + iy * nx + ix1] * (x1 - ix1) * (iz2 + 1 - z2);
	integral += buff[iz2 * nx * ny + iy * nx + ix2] * (ix2 + 1 - x2) * (iz2 + 1 - z2);

	return integral;

}


// Calculate the integral value inside a box
// integral = (pixel sum) - (edge leftout) + (corner leftout)
__device__ float IntegralBoxYZ(
	const float* buff, float y1, float z1, float y2, float z2, const int ix, size_t nx, size_t ny, size_t nz
)
{
	y1 = ClampFloat(y1, 0, ny);
	z1 = ClampFloat(z1, 0, nz);
	y2 = ClampFloat(y2, 0, ny);
	z2 = ClampFloat(z2, 0, nz);

	int iy1 = Clamp(int(y1), 0, ny);
	int iz1 = Clamp(int(z1), 0, nz);
	int iy2 = Clamp(int(y2), 0, ny);
	int iz2 = Clamp(int(z2), 0, nz);

	// there is not area to integral on
	if (z2 <= z1 || y2 <= y1)
	{
		return 0;
	}

	float integral = 0;	

	// 1. Calculate the integral of all the pixels
	for (int iy = iy1; iy <= iy2; iy++)
	{
		for (int iz = iz1; iz <= iz2; iz++)
		{
			integral += buff[iz * nx * ny + iy * nx + ix];
		}
	}

	// 2. subtract the integral on the 4 sides
	// left
	for (int iy = iy1; iy <= iy2; iy++)
	{
		integral -= buff[iz1 * nx * ny + iy * nx + ix] * (z1 - iz1);
	}
	// right
	for (int iy = iy1; iy <= iy2; iy++)
	{
		integral -= buff[iz2 * nx * ny + iy * nx + ix] * (iz2 + 1 - z2);
	}
	// top
	for (int iz = iz1; iz <= iz2; iz++)
	{
		integral -= buff[iz * nx * ny + iy1 * nx + ix] * (y1 - iy1);
	}
	// bottom
	for (int iz = iz1; iz <= iz2; iz++)
	{
		integral -= buff[iz * nx * ny + iy2 * nx + ix] * (iy2 + 1 - y2);
	}

	// 3. add the corner integrals
	integral += buff[iz1 * nx * ny + iy1 * nx + ix] * (z1 - iz1) * (y1 - iy1);
	integral += buff[iz2 * nx * ny + iy1 * nx + ix] * (iz2 + 1 - z2) * (y1 - iy1);
	integral += buff[iz1 * nx * ny + iy2 * nx + ix] * (z1 - iz1) * (iy2 + 1 - y2);
	integral += buff[iz2 * nx * ny + iy2 * nx + ix] * (iz2 + 1 - z2) * (iy2 + 1 - y2);

	return integral;

}

// Calculate the integral value inside a box in 2D
// integral = (pixel sum) - (leftout)
__device__ float IntegralBoxX(
	const float* buff, float x1, float x2, const int iy, const int iz, size_t nx, size_t ny, size_t nz
)
{
	x1 = ClampFloat(x1, 0, nx);
	x2 = ClampFloat(x2, 0, nx);

	int ix1 = Clamp(int(x1), 0, nx);
	int ix2 = Clamp(int(x2), 0, nx);

	// there is not area to integral on
	if (x2 <= x1)
	{
		return 0;
	}

	float integral = 0;	

	// 1. Calculate the integral of all the pixels
	for (int ix = ix1; ix <= ix2; ix++)
	{
		integral += buff[iz * nx * ny + iy * nx + ix];
	}

	// 2. subtract the integral on the 2 sides 
	// left
	integral -= buff[iz * nx * ny + iy * nx + ix1] * (x1 - ix1);
	// right
	integral -= buff[iz * nx * ny + iy * nx + ix2] * (ix2 + 1 - x2);

	return integral;

}


// Calculate the integral value inside a box in 2D
// integral = (pixel sum) - (leftout)
__device__ float IntegralBoxY(
	const float* buff, float y1, float y2, const int ix, const int iz, size_t nx, size_t ny, size_t nz
)
{
	y1 = ClampFloat(y1, 0, ny);
	y2 = ClampFloat(y2, 0, ny);

	int iy1 = Clamp(int(y1), 0, ny);
	int iy2 = Clamp(int(y2), 0, ny);

	// there is not area to integral on
	if (y2 <= y1)
	{
		return 0;
	}

	float integral = 0;	

	// 1. Calculate the integral of all the pixels
	for (int iy = iy1; iy <= iy2; iy++)
	{
		integral += buff[iz * nx * ny + iy * nx + ix];
	}

	// 2. subtract the integral on the 2 sides 
	// top
	integral -= buff[iz * nx * ny + iy1 * nx + ix] * (y1 - iy1);
	// bottom
	integral -= buff[iz * nx * ny + iy2 * nx + ix] * (iy2 + 1 - y2);

	return integral;

}