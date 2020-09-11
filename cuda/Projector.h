#pragma once

#include <stdio.h>

#define PI 3.141592657f

struct Grid
{
	int nx;
	int ny;
	int nz;
	float dx;
	float dy;
	float dz;
	float cx;
	float cy;
	float cz;
};

struct Detector
{
	int nu;
	int nv;
	float du;
	float dv;
	float off_u;
	float off_v;
};

inline __device__ __host__ Grid MakeGrid(int nx, int ny, int nz, float dx, float dy, float dz,
		float cx, float cy, float cz)
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

inline __device__ __host__ Detector MakeDetector(int nu, int nv, float du, float dv, float off_u, float off_v)
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

class Projector
{
public:
	int nBatches;		// number of batches

	// image parameters
	int nx;
	int ny;
	int nz;
	float dx; // mm
	float dy; // mm
	float dz; // mm
	float cx; // center of image in mm
	float cy;
	float cz;

	// sinogram parameters
	int nu;
	int nview;
	int nv;
	float du;
	float dv;
	float off_u;
	float off_v;

	// geometries
	float dsd;
	float dso;

	int typeProjector; // to tag slight changes between different versions, e.g. BP for FBP do not need length factor


public:
	void Setup(int nBatches, 
		int nx, int ny, int nz, float dx, float dy, float dz, float cx, float cy, float cz,
		int nu, int nview, int nv, float du, float dv, float off_u, float off_v,
		float dsd, float dso, int typeProjector = 0);

public:
	void SetCudaStream(const cudaStream_t& stream);

public:
	virtual void Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg) {};
	virtual void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) {};

protected:
	cudaStream_t m_stream;

public:
	Projector();
	virtual ~Projector();

};

