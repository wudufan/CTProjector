#include "projector.h"
#include "siddon.h"
#include "cudaMath.h"

#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace std;

#define eps 1e-6f

__device__ bool InboundAlpha(float& alpha0v, float& alphaNv, float dDstSrcv, float srcv, int gridnv)
{
	if (fabsf(dDstSrcv) > eps)
	{
		// non-parallel along x
		alpha0v = (0 - srcv) / dDstSrcv;
		alphaNv = (gridnv - srcv) / dDstSrcv;
	}
	else
	{
		// parallel along x
		if (srcv < 0 || srcv > gridnv)
		{
			// no intersection
			return false;
		}
	}

	return true;
}

// the first intersection of the ray with the grid, given separately for x, y, and z
__device__ int InboundFirstVoxel(float pt0v, float dDstSrcv, int gridnv)
{
	int ngv = (int)pt0v;
	if (fabs(dDstSrcv) > eps)
	{
		if (ngv < 0)
		{
			ngv = 0;
		}
		else if (ngv >= gridnv)
		{
			ngv = gridnv - 1;
		}
	}

	return ngv;
}

// the second intersection of the raw with the grid, gien seperately for x, y, and z
__device__ float OutboundFirstVoxel(int ngv, float dDstSrcv)
{
	if (dDstSrcv > eps)
	{
		return (float)(ngv + 1);
	}
	else if (dDstSrcv < eps)
	{
		return (float)(ngv);
	}
	else
	{
		return (float)(ngv);
	}
}

// get the alpha of the second intersection and the dAlpha along each direction
__device__ void GetAlpha(float& alphav, float& dAlphav, float pt1v, float srcv, float dDstSrcv)
{
	if (fabsf(dDstSrcv) > eps)
	{
		alphav = (pt1v - srcv) / dDstSrcv;
		dAlphav = 1 / fabsf(dDstSrcv);
	}
}

__device__ SiddonTracingVars InitializeSiddon(float3 src, float3 dst, const Grid grid)
{
	SiddonTracingVars var;

	float dist = sqrtf(
		(src.x - dst.x) * (src.x - dst.x)
		+ (src.y - dst.y) * (src.y - dst.y)
		+ (src.z - dst.z) * (src.z - dst.z)
	);

	src = PhysicsToImg(src, grid);
	dst = PhysicsToImg(dst, grid);

	float3 dDstSrc = make_float3(dst.x - src.x, dst.y - src.y, dst.z - src.z);

	// intersection between ray and grid
	float3 a0 = make_float3(1.0f, 1.0f, 1.0f);
	float3 a1 = make_float3(0.0f, 0.0f, 0.0f);
	if (!InboundAlpha(a0.x, a1.x, dDstSrc.x, src.x, grid.nx)
		|| !InboundAlpha(a0.y, a1.y, dDstSrc.y, src.y, grid.ny) 
		|| !InboundAlpha(a0.z, a1.z, dDstSrc.z, src.z, grid.nz))
	{
		var.alpha.x = -1;
		return var;
	}

	// entry and exit points
	float amin = fmaxf(0.f, fmaxf(fminf(a0.x, a1.x), fmaxf(fminf(a0.y, a1.y), fminf(a0.z, a1.z))));	// entry
	float amax = fminf(1.f, fminf(fmaxf(a0.x, a1.x), fminf(fmaxf(a0.y, a1.y), fmaxf(a0.z, a1.z))));	// exit

	if (amax <= amin)
	{
		// no intersection
		var.alpha.x = -1;
		return var;
	}

	// entry point
	float3 pt0 = make_float3(src.x + amin * dDstSrc.x, src.y + amin * dDstSrc.y, src.z + amin * dDstSrc.z);

	// first intersection voxel
	int3 ng0 = make_int3(
		InboundFirstVoxel(pt0.x, dDstSrc.x, grid.nx),
		InboundFirstVoxel(pt0.y, dDstSrc.y, grid.ny),
		InboundFirstVoxel(pt0.z, dDstSrc.z, grid.nz)
	);

	// exiting point from first voxel
	float3 pt1 = make_float3(
		OutboundFirstVoxel(ng0.x, dDstSrc.x),
		OutboundFirstVoxel(ng0.y, dDstSrc.y),
		OutboundFirstVoxel(ng0.z, dDstSrc.z)
	);

	// the alpha of the exiting point and step size along each direction
	float3 alpha = make_float3(2.0f, 2.0f, 2.0f);	// the max value of alpha is 1, so if alpha is not set in any direction then it will be skipped
	float3 dAlpha = make_float3(0.0f, 0.0f, 0.0f);
	GetAlpha(alpha.x, dAlpha.x, pt1.x, src.x, dDstSrc.x);
	GetAlpha(alpha.y, dAlpha.y, pt1.y, src.y, dDstSrc.y);
	GetAlpha(alpha.z, dAlpha.z, pt1.z, src.z, dDstSrc.z);

	var.alpha = make_float3(alpha.x * dist, alpha.y * dist, alpha.z * dist);
	var.dAlpha = make_float3(dAlpha.x * dist, dAlpha.y * dist, dAlpha.z * dist);
	var.ng = ng0;
	var.isPositive = make_int3((dDstSrc.x > 0) ? 1 : 0, (dDstSrc.y > 0) ? 1 : 0, (dDstSrc.z > 0) ? 1 : 0);
	var.alphaNow = var.alphaPrev = amin * dist;
	var.rayLength = dist * (amax - amin);

	return var;
}

__device__ float SiddonRayTracing(float* pPrj, const float* pImg, float3 src, float3 dst, const Grid grid)
{
	SiddonTracingVars var = InitializeSiddon(src, dst, grid);

	if (var.alpha.x < -0.5 || var.ng.x < 0 || var.ng.x >= grid.nx
		|| var.ng.y < 0 || var.ng.y >= grid.ny
		|| var.ng.z < 0 || var.ng.z >= grid.nz)
	{
		// no intersections
		return 0;
	}

	int move = 0;
	bool isTracing = true;
	float val = 0;
	int nxy = grid.nx * grid.ny;
	pImg += var.ng.z * nxy + var.ng.y * grid.nx + var.ng.x;

	while (isTracing)
	{
		// each iteration find the direction of alpha nearest to the src,
		// set it to alphaNow then move it to the next intersection with grid along that direction
		if (var.alpha.x < var.alpha.y && var.alpha.x < var.alpha.z)
		{
			var.alphaNow = var.alpha.x;
			var.alpha.x += var.dAlpha.x;
			if (var.isPositive.x == 1)
			{
				var.ng.x++;
				move = 1;
				if (var.ng.x >= grid.nx) isTracing = false;
			}
			else
			{
				var.ng.x--;
				move = -1;
				if (var.ng.x < 0) isTracing = false;
			}
		}
		else if (var.alpha.y < var.alpha.z)
		{
			var.alphaNow = var.alpha.y;
			var.alpha.y += var.dAlpha.y;
			if (var.isPositive.y == 1)
			{
				var.ng.y++;
				move = grid.nx;
				if (var.ng.y >= grid.ny) isTracing = false;
			}
			else
			{
				var.ng.y--;
				move = -grid.nx;
				if (var.ng.y < 0) isTracing = false;
			}
		}
		else
		{
			var.alphaNow = var.alpha.z;
			var.alpha.z += var.dAlpha.z;
			if (var.isPositive.z == 1)
			{
				var.ng.z++;
				move = nxy;
				if (var.ng.z >= grid.nz) isTracing = false;
			}
			else
			{
				var.ng.z--;
				move = -nxy;
				if (var.ng.z < 0) isTracing = false;
			}
		}

		val += (*pImg) * (var.alphaNow - var.alphaPrev);
		var.alphaPrev = var.alphaNow;
		pImg += move;
	}

	*pPrj += val;

	return var.rayLength;
}

__device__ float SiddonRayTracingTransposeAtomicAdd(float* pImg, float val, float3 src, float3 dst, const Grid grid)
{
	SiddonTracingVars var = InitializeSiddon(src, dst, grid);

	if (var.alpha.x < -0.5 || var.ng.x < 0 || var.ng.x >= grid.nx
			|| var.ng.y < 0 || var.ng.y >= grid.ny
			|| var.ng.z < 0 || var.ng.z >= grid.nz)
	{
		// no intersections
		return 0;
	}

	int move = 0;
	bool isTracing = true;
	int nxy = grid.nx * grid.ny;
	pImg += var.ng.z * nxy + var.ng.y * grid.nx + var.ng.x;

	while (isTracing)
	{
		// each iteration find the direction of alpha nearest to the src,
		// set it to alphaNow then move it to the next intersection with grid along that direction
		if (var.alpha.x < var.alpha.y && var.alpha.x < var.alpha.z)
		{
			var.alphaNow = var.alpha.x;
			var.alpha.x += var.dAlpha.x;
			if (var.isPositive.x == 1)
			{
				var.ng.x++;
				move = 1;
				if (var.ng.x >= grid.nx) isTracing = false;
			}
			else
			{
				var.ng.x--;
				move = -1;
				if (var.ng.x < 0) isTracing = false;
			}
		}
		else if (var.alpha.y < var.alpha.z)
		{
			var.alphaNow = var.alpha.y;
			var.alpha.y += var.dAlpha.y;
			if (var.isPositive.y == 1)
			{
				var.ng.y++;
				move = grid.nx;
				if (var.ng.y >= grid.ny) isTracing = false;
			}
			else
			{
				var.ng.y--;
				move = -grid.nx;
				if (var.ng.y < 0) isTracing = false;
			}
		}
		else
		{
			var.alphaNow = var.alpha.z;
			var.alpha.z += var.dAlpha.z;
			if (var.isPositive.z == 1)
			{
				var.ng.z++;
				move = nxy;
				if (var.ng.z >= grid.nz) isTracing = false;
			}
			else
			{
				var.ng.z--;
				move = -nxy;
				if (var.ng.z < 0) isTracing = false;
			}
		}

		atomicAdd(pImg, val * (var.alphaNow - var.alphaPrev));
		var.alphaPrev = var.alphaNow;
		pImg += move;
	}

	return var.rayLength;
}


// move source and dst near the grid to avoid precision problems in further ray tracing
__device__ __host__ void MoveSourceDstNearGrid(float3& src, float3&dst, const Grid grid)
{
	// twice the radius of grid
	float r = vecLength(make_float3(grid.nx * grid.dx, grid.ny * grid.dy, grid.nz * grid.dz));
	// dst to src direction
	float3 d = vecNormalize(vecMinus(dst, src));
	// distance from source to central plane
	float t0 = vecDot(make_float3(grid.cx - src.x, grid.cy - src.y, grid.cz - src.z), d);
	// distance from dst to central plane
	float t1 = vecDot(make_float3(dst.x - grid.cx, dst.y - grid.cy, dst.z - grid.cz), d);

	if (t0 > r)
	{
		src = make_float3(src.x + (t0 - r) * d.x, src.y + (t0 - r) * d.y, src.z + (t0 - r) * d.z);
	}

	if (t1 > r)
	{
		dst = make_float3(dst.x - (t1 - r) * d.x, dst.y - (t1 - r) * d.y, dst.z - (t1 - r) * d.z);
	}
}
