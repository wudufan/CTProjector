#include "distanceDriven.h"
#include "cudaMath.h"
#include "projector.h"

#include <stdexcept>
#include <exception>
#include <sstream>
#include <iostream>

using namespace std;

bool DistanceDrivenFan::isFBP() {
	int typeProjectorMasked = typeProjector & 1;
	return (typeProjectorMasked == 1);
}

bool DistanceDrivenFan::isBranchless() {
	int typeProjectorMasked = typeProjector & 2;
	return (typeProjectorMasked == 2);
}

bool DistanceDrivenFan::forceFBPWeight() {
	int typeProjectorMasked = typeProjector & 4;
	return (typeProjectorMasked == 4);
}

void DistanceDrivenFan::Allocate(bool forward, bool backward)
{
	this->Free();
	Projector::Allocate(forward, backward);

	if (forward)
	{
		if (isBranchless()) {
			if (cudaSuccess != cudaMalloc(&pAccX, sizeof(float) * (nx + 1) * ny * nz))
			{
				throw runtime_error("pAccX allocation failed");
			}
			if (cudaSuccess != cudaMalloc(&pAccY, sizeof(float) * nx * (ny + 1) * nz))
			{
				throw runtime_error("pAccY allocation failed");
			}
		}
	}
	if (backward)
	{
		if (cudaSuccess != cudaMalloc(&pWeightedPrjs, sizeof(float) * nu * nv * nview))
		{
			throw runtime_error("pWeightedPrjs allocation failed");
		}
		if (isBranchless()) {
			if (cudaSuccess != cudaMalloc(&pAccU, sizeof(float) * (nu + 1) * nv * nview))
			{
				throw runtime_error("pAccU allocation failed");
			}
		}

	}
	
}

void DistanceDrivenFan::AllocateExternal(float* pAccXEx, float* pAccYEx, float* pAccUEx, float* pWeightedPrjsEx) {
	this->Free();
	Projector::AllocateExternal();

	pAccX = pAccXEx;
	pAccY = pAccYEx;
	pAccU = pAccUEx;
	pWeightedPrjs = pWeightedPrjsEx;
	
}

void DistanceDrivenFan::Free()
{
	if (!m_externalBuffer) {
		if (pAccX != NULL) cudaFree(pAccX);
		if (pAccY != NULL) cudaFree(pAccY);
		if (pAccU != NULL) cudaFree(pAccU);
		if (pWeightedPrjs != NULL) cudaFree(pWeightedPrjs);
	}

	pAccX = NULL;
	pAccY = NULL;
	pAccU = NULL;
	pWeightedPrjs = NULL;
}

void DistanceDrivenFan::Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg)
{
	if (isBranchless()) {
		ProjectionBranchless(pcuImg, pcuPrj, pcuDeg);
	}
	else {
		ProjectionBoxInt(pcuImg, pcuPrj, pcuDeg);
	}

}

void DistanceDrivenParallel::Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg)
{
	if (isBranchless()) {
		ProjectionBranchless(pcuImg, pcuPrj, pcuDeg);
	}
	else {
		ProjectionBoxInt(pcuImg, pcuPrj, pcuDeg);
	}
}

extern "C" int cupyDistanceDrivenFanProjection(
	float* prj,
	const float* img,
	const float* deg,
	size_t nBatches,
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
	float da,
	float dv,
	float off_a,
	float off_v,
	float dsd,
	float dso,
	int typeProjector
)
{
	try
	{
		DistanceDrivenFan projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, da, dv, off_a, off_v, dsd, dso, typeProjector
		);
		
		projector.Allocate(true, false);
		projector.Projection(img, prj, deg);
		projector.Free();
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cupyDistanceDrivenFanProjection() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();
}

extern "C" int cupyDistanceDrivenParallelProjection(
	float* prj,
	const float* img,
	const float* deg,
	size_t nBatches,
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
	float da,
	float dv,
	float off_a,
	float off_v,
	int typeProjector
)
{
	try
	{
		DistanceDrivenParallel projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, da, dv, off_a, off_v, 1000, 500, typeProjector
		);
		
		projector.Allocate(true, false);
		projector.Projection(img, prj, deg);
		projector.Free();
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cupyDistanceDrivenParallelProjection() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();
}

extern "C" int cDistanceDrivenFanProjection(
	float* prj,
	const float* img,
	const float* deg,
	size_t nBatches,
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
	float da,
	float dv,
	float off_a,
	float off_v,
	float dsd,
	float dso,
	int typeProjector
)
{
	float* pcuImg = NULL;
	float* pcuPrj = NULL;
	float* pcuDeg = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nBatches * nx * ny * nz))
		{
			throw runtime_error("pcuImg allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nu * nv * nview))
		{
			throw runtime_error("pcuPrj allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuDeg, sizeof(float) * nview))
		{
			throw runtime_error("pcuDeg allocation failed");
		}

		cudaMemcpy(pcuDeg, deg, sizeof(float) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuImg, img, sizeof(float) * nBatches * nx * ny * nz, cudaMemcpyHostToDevice);
		cudaMemset(pcuPrj, 0, sizeof(float) * nBatches * nu * nv * nview);

		DistanceDrivenFan projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, da, dv, off_a, off_v, dsd, dso, typeProjector
		);
		
		projector.Allocate(true, false);
		projector.Projection(pcuImg, pcuPrj, pcuDeg);
		projector.Free();

		cudaMemcpy(prj, pcuPrj, sizeof(float) * nBatches * nu * nv * nview, cudaMemcpyDeviceToHost);

	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cDistanceDrivenFanProjection() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	if (pcuImg != NULL) cudaFree(pcuImg);
	if (pcuPrj != NULL) cudaFree(pcuPrj);
	if (pcuDeg != NULL) cudaFree(pcuDeg);

	return cudaGetLastError();
}


extern "C" int cDistanceDrivenParallelProjection(
	float* prj,
	const float* img,
	const float* deg,
	size_t nBatches,
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
	float da,
	float dv,
	float off_a,
	float off_v,
	int typeProjector
)
{
	float* pcuImg = NULL;
	float* pcuPrj = NULL;
	float* pcuDeg = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nBatches * nx * ny * nz))
		{
			throw runtime_error("pcuImg allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nu * nv * nview))
		{
			throw runtime_error("pcuPrj allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuDeg, sizeof(float) * nview))
		{
			throw runtime_error("pcuDeg allocation failed");
		}

		cudaMemcpy(pcuDeg, deg, sizeof(float) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuImg, img, sizeof(float) * nBatches * nx * ny * nz, cudaMemcpyHostToDevice);
		cudaMemset(pcuPrj, 0, sizeof(float) * nBatches * nu * nv * nview);

		DistanceDrivenParallel projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, da, dv, off_a, off_v, 1000, 500, typeProjector
		);
		
		projector.Allocate(true, false);
		projector.Projection(pcuImg, pcuPrj, pcuDeg);
		projector.Free();

		cudaMemcpy(prj, pcuPrj, sizeof(float) * nBatches * nu * nv * nview, cudaMemcpyDeviceToHost);
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cDistanceDrivenParallelProjection() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	if (pcuImg != NULL) cudaFree(pcuImg);
	if (pcuPrj != NULL) cudaFree(pcuPrj);
	if (pcuDeg != NULL) cudaFree(pcuDeg);

	return cudaGetLastError();
}


void DistanceDrivenFan::Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg)
{
	if (isBranchless()) {
		BackprojectionBranchless(pcuImg, pcuPrj, pcuDeg);
	}
	else {
		BackprojectionBoxInt(pcuImg, pcuPrj, pcuDeg);
	}

}

void DistanceDrivenParallel::Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg)
{
	if (isBranchless()) {
		BackprojectionBranchless(pcuImg, pcuPrj, pcuDeg);
	}
	else {
		BackprojectionBoxInt(pcuImg, pcuPrj, pcuDeg);
	}

}

extern "C" int cupyDistanceDrivenFanBackprojection(
	float* img,
	const float* prj,
	const float* deg,
	size_t nBatches, 
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
	float da,
	float dv,
	float off_a,
	float off_v,
	float dsd,
	float dso,
	int typeProjector
)
{
	try
	{
		DistanceDrivenFan projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz, nu, nv, nview, da, dv, off_a, off_v, dsd, dso, typeProjector
		);
		
		projector.Allocate(false, true);
		projector.Backprojection(img, prj, deg);		
		projector.Free();
	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "cupyDistanceDrivenFanBackprojection failed: " << e.what()
			<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();

}


extern "C" int cupyDistanceDrivenParallelBackprojection(
	float* img,
	const float* prj,
	const float* deg,
	size_t nBatches, 
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
	float da,
	float dv,
	float off_a,
	float off_v,
	int typeProjector
)
{
	try
	{
		DistanceDrivenParallel projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz, nu, nv, nview, da, dv, off_a, off_v, 1000, 500, typeProjector
		);
		
		projector.Allocate(false, true);
		projector.Backprojection(img, prj, deg);		
		projector.Free();
	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "cupyDistanceDrivenParallelBackprojection failed: " << e.what()
			<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();

}


extern "C" int cDistanceDrivenFanBackprojection(
	float* img,
	const float* prj,
	const float* deg,
	size_t nBatches, 
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
	float da,
	float dv,
	float off_a,
	float off_v,
	float dsd,
	float dso,
	int typeProjector
)
{
	float* pcuImg = NULL;
	float* pcuPrj = NULL;
	float* pcuDeg = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nBatches * nz * ny * nx))
		{
			throw runtime_error("pcuImg allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nview * nv * nu))
		{
			throw runtime_error("pcuPrj allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuDeg, sizeof(float) * nview))
		{
			throw runtime_error("pcuDeg allocation failed");
		}

		cudaMemcpy(pcuDeg, deg, sizeof(float) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuPrj, prj, sizeof(float) * nBatches * nview * nv * nu, cudaMemcpyHostToDevice);
		cudaMemset(pcuImg, 0, sizeof(float) * nBatches * nz * ny * nx);

		DistanceDrivenFan projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz, nu, nv, nview, da, dv, off_a, off_v, dsd, dso, typeProjector
		);
		
		projector.Allocate(false, true);
		projector.Backprojection(pcuImg, pcuPrj, pcuDeg);
		projector.Free();

		cudaMemcpy(img, pcuImg, sizeof(float) * nBatches * nz * ny * nx, cudaMemcpyDeviceToHost);
	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "cDistanceDrivenFanBackprojection failed: " << e.what()
			<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	if (pcuImg != NULL) cudaFree(pcuImg);
	if (pcuPrj != NULL) cudaFree(pcuPrj);
	if (pcuDeg != NULL) cudaFree(pcuDeg);

	return cudaGetLastError();

}

extern "C" int cDistanceDrivenParallelBackprojection(
	float* img,
	const float* prj,
	const float* deg,
	size_t nBatches, 
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
	float da,
	float dv,
	float off_a,
	float off_v,
	int typeProjector
)
{
	float* pcuImg = NULL;
	float* pcuPrj = NULL;
	float* pcuDeg = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nBatches * nz * ny * nx))
		{
			throw runtime_error("pcuImg allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nview * nv * nu))
		{
			throw runtime_error("pcuPrj allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuDeg, sizeof(float) * nview))
		{
			throw runtime_error("pcuDeg allocation failed");
		}

		cudaMemcpy(pcuDeg, deg, sizeof(float) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuPrj, prj, sizeof(float) * nBatches * nview * nv * nu, cudaMemcpyHostToDevice);
		cudaMemset(pcuImg, 0, sizeof(float) * nBatches * nz * ny * nx);

		DistanceDrivenParallel projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz, nu, nv, nview, da, dv, off_a, off_v, 1000, 500, typeProjector
		);
		
		projector.Allocate(false, true);
		projector.Backprojection(pcuImg, pcuPrj, pcuDeg);		
		projector.Free();

		cudaMemcpy(img, pcuImg, sizeof(float) * nBatches * nz * ny * nx, cudaMemcpyDeviceToHost);
	}
	catch (exception &e)
	{
		ostringstream oss;
		oss << "cDistanceDrivenParallelBackprojection failed: " << e.what()
			<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	if (pcuImg != NULL) cudaFree(pcuImg);
	if (pcuPrj != NULL) cudaFree(pcuPrj);
	if (pcuDeg != NULL) cudaFree(pcuDeg);

	return cudaGetLastError();

}
