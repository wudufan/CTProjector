#include "projector.h"

#include <exception>
#include <stdexcept>
#include <cuda_runtime_api.h>

#include <iostream>
#include <sstream>

using namespace std;

extern "C" int SetDevice(int device)
{
	return cudaSetDevice(device);
}

Projector::Projector()
{
	m_stream = NULL;
	m_externalBuffer = false;

	nBatches = 0;

	nx = 0;
	ny = 0;
	nz = 0;
	dx = 0;
	dy = 0;
	dz = 0;
	cx = 0;
	cy = 0;
	cz = 0;

	nu = 0;
	nview = 0;
	nv = 0;
	du = 0;
	dv = 0;
	off_u = 0;
	off_v = 0;

	dsd = 0;
	dso = 0;

	typeProjector = 0;

}

Projector::~Projector()
{
	this->Free();
}

void Projector::SetCudaStream(const cudaStream_t& stream)
{
	m_stream = stream;
}

void Projector::Setup(
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
	int typeProjector
)
{
	this->nBatches = nBatches;

	this->nx = nx;
	this->ny = ny;
	this->nz = nz;
	this->dx = dx;
	this->dy = dy;
	this->dz = dz;
	this->cx = cx;
	this->cy = cy;
	this->cz = cz;

	this->nu = nu;
	this->nview = nview;
	this->nv = nv;
	this->du = du;
	this->dv = dv;
	this->off_u = off_u;
	this->off_v = off_v;

	this->dsd = dsd;
	this->dso = dso;

	this->typeProjector = typeProjector;
}
