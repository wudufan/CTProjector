#pragma once

#include <stdio.h>

#define PI 3.141592657f

struct Grid
{
	size_t nx;
	size_t ny;
	size_t nz;
	float dx;
	float dy;
	float dz;
	float cx;
	float cy;
	float cz;
};

struct Detector
{
	size_t nu;
	size_t nv;
	float du;
	float dv;
	float off_u;
	float off_v;
};

inline __device__ __host__ Grid MakeGrid(
	size_t nx, size_t ny, size_t nz, float dx, float dy, float dz, float cx, float cy, float cz
)
{
	Grid grid;
	grid.nx = nx;
	grid.ny = ny;
	grid.nz = nz;
	grid.dx = dx;
	grid.dy = dy;
	grid.dz = dz;
	grid.cx = cx;
	grid.cy = cy;
	grid.cz = cz;

	return grid;
}

inline __device__ __host__ Detector MakeDetector(
	size_t nu, size_t nv, float du, float dv, float off_u, float off_v
)
{
	Detector det;
	det.nu = nu;
	det.nv = nv;
	det.du = du;
	det.dv = dv;
	det.off_u = off_u;
	det.off_v = off_v;

	return det;
}

// The original point of the image is defined as the corner of the first pixel under this conversion 
inline __device__ float3 PhysicsToImg(float3 pt, const Grid grid)
{
	pt.x = (pt.x - grid.cx) / grid.dx + grid.nx / 2.0f;
	pt.y = (pt.y - grid.cy) / grid.dy + grid.ny / 2.0f;
	pt.z = (pt.z - grid.cz) / grid.dz + grid.nz / 2.0f;

	return pt;
}

inline __device__ float2 PhysicsToImg(float2 pt, const Grid grid)
{
	pt.x = (pt.x - grid.cx) / grid.dx + grid.nx / 2.0f;
	pt.y = (pt.y - grid.cy) / grid.dy + grid.ny / 2.0f;

	return pt;
}

inline __device__ float3 ImgToPhysics(float3 pt, const Grid grid)
{
	pt.x = (pt.x - grid.nx / 2.f) * grid.dx + grid.cx;
	pt.y = (pt.y - grid.ny / 2.f) * grid.dy + grid.cy;
	pt.z = (pt.z - grid.nz / 2.f) * grid.dz + grid.cz;

	return pt;
}


inline __device__ float2 ImgToPhysics(float2 pt, const Grid grid)
{
	pt.x = (pt.x - grid.nx / 2.f) * grid.dx + grid.cx;
	pt.y = (pt.y - grid.ny / 2.f) * grid.dy + grid.cy;

	return pt;
}


class Projector
{
public:
	size_t nBatches;		// number of batches

	// image parameters
	size_t nx;
	size_t ny;
	size_t nz;
	float dx; // mm
	float dy; // mm
	float dz; // mm
	float cx; // center of image in mm
	float cy;
	float cz;

	// sinogram parameters
	size_t nu;
	size_t nview;
	size_t nv;
	float du;
	float dv;
	float off_u;
	float off_v;

	// geometries
	float dsd;
	float dso;

	int typeProjector; // to tag slight changes between different versions, e.g. BP for FBP do not need length factor


public:
	void Setup(
		int nBatches, 
		size_t nx,
		size_t ny,
		size_t nz,
		float dx,
		float dy,
		float dz,
		float cx,
		float cy,
		float cz,
		size_t nu,
		size_t nv,
		size_t nview,
		float du,
		float dv,
		float off_u,
		float off_v,
		float dsd,
		float dso,
		int typeProjector = 0
	);

public:
	void SetCudaStream(const cudaStream_t& stream);

public:
	// allocate all the needed memory in advance for tensorflow compatibility
	virtual void Allocate(bool forward = true, bool backward = true) { m_externalBuffer = false; }
	void AllocateExternal() { m_externalBuffer = true; }
	virtual void Free() {};

public:
	virtual void Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg) {};
	virtual void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) {};

protected:
	cudaStream_t m_stream;
	bool m_externalBuffer;

public:
	Projector();
	virtual ~Projector();

};

