/*
C-interface for distanceDriven3D.cu
*/

#include "distanceDriven.h"

#include <stdexcept>
#include <exception>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;


void DistanceDrivenTomo::ProjectionTomo(const float* pcuImg, float* pcuPrj, const float* pcuDetCenter, const float* pcuSrc)
{
	// allocate buffers
	double* pcuAcc = NULL;
	int* pcuIviews = NULL;

	float3* pcuDetU = NULL;
	float3* pcuDetV = NULL;

	try
	{	
		if (typeProjector == 1)
		{
			if (cudaSuccess != cudaMalloc(&pcuAcc, sizeof(double) * (nx + 1) * (ny + 1) * nz))
			{
				throw runtime_error("pAcc allocation failed");
			}
		}

		if (cudaSuccess != cudaMalloc(&pcuDetU, sizeof(float3) * nview))
		{
			throw runtime_error("pDetU allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuDetV, sizeof(float3) * nview))
		{
			throw runtime_error("pDetV allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuIviews, sizeof(int) * nview))
		{
			throw runtime_error("cuIviews allocation failed");
		}

		// cuIviews should contain all angles
		int* iviews = new int [nview];
		for (int i = 0; i < nview; i++)
		{
			iviews[i] = i;
		}
		cudaMemcpy(pcuIviews, iviews, sizeof(int) * nview, cudaMemcpyHostToDevice);
		delete [] iviews;

		// pcuDetU should contain all (1,0,0)
		// pcuDetV should contain all (0,1,0)
		float3* pDetU = new float3 [nview];
		for (int i = 0; i < nview; i++)
		{
			pDetU[i] = make_float3(1, 0, 0);
		}
		cudaMemcpy(pcuDetU, pDetU, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		delete [] pDetU;

		float3* pDetV = new float3 [nview];
		for (int i = 0; i < nview; i++)
		{
			pDetV[i] = make_float3(0, 1, 0);
		}
		cudaMemcpy(pcuDetV, pDetV, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		delete [] pDetV;

		if (typeProjector == 1)
		{
			ProjectionTomoBranchless(
				pcuImg, pcuPrj, pcuDetCenter, pcuSrc, pcuAcc, pcuIviews, pcuDetU, pcuDetV
			);
		}
		else
		{
			ProjectionTomoBoxInt(
				pcuImg, pcuPrj, pcuDetCenter, pcuSrc, pcuIviews, pcuDetU, pcuDetV
			);
		}
	}
	catch (exception &e)
	{
		if (pcuAcc != NULL) cudaFree(pcuAcc);
		if (pcuIviews != NULL) cudaFree(pcuIviews);
		if (pcuDetU != NULL) cudaFree(pcuDetU);
		if (pcuDetV != NULL) cudaFree(pcuDetV);

		ostringstream oss;
		oss << "DistanceDrivenTomo::ProjectionTomo Error: " << e.what() 
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}

	if (pcuAcc != NULL) cudaFree(pcuAcc);
	if (pcuIviews != NULL) cudaFree(pcuIviews);
	if (pcuDetU != NULL) cudaFree(pcuDetU);
	if (pcuDetV != NULL) cudaFree(pcuDetV);

}

// no textures, use double-precision software interpolation
void DistanceDrivenTomo::BackprojectionTomo(float* pcuImg, const float* pcuPrj, const float* pcuDetCenter, const float* pcuSrc)
{
	// the backprojection is constrained to Cartesian coordinate for simplification, hence no detU / detV needed
	float* pcuWeightedPrjs = NULL;
	double* pcuAcc = NULL;
	int* pcuIviews = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuWeightedPrjs, sizeof(float) * nBatches * nu * nv * nview))
		{
			throw runtime_error("pWeightedPrjs allocation failed");
		}

		if (typeProjector == 1)
		{
			if (cudaSuccess != cudaMalloc(&pcuAcc, sizeof(double) * (nu + 1) * (nv + 1) * nview))
			{
				throw runtime_error("pAcc allocation failed");
			}
		}

		if (cudaSuccess != cudaMalloc(&pcuIviews, sizeof(int) * nview))
		{
			throw runtime_error("cuIviews allocation failed");
		}

		// cuIviews should contain all angles
		int* iviews = new int [nview];
		for (int i = 0; i < nview; i++)
		{
			iviews[i] = i;
		}
		cudaMemcpy(pcuIviews, iviews, sizeof(int) * nview, cudaMemcpyHostToDevice);
		delete [] iviews;

		if (typeProjector == 1)
		{
			BackprojectionTomoBranchless(
				pcuImg, pcuPrj, pcuDetCenter, pcuSrc, pcuWeightedPrjs, pcuAcc, pcuIviews
			);
		}
		else
		{
			BackprojectionTomoBoxInt(
				pcuImg, pcuPrj, pcuDetCenter, pcuSrc, pcuWeightedPrjs, pcuIviews
			);
		}

	}
	catch (exception &e)
	{
		if (pcuWeightedPrjs != NULL) cudaFree(pcuWeightedPrjs);
		if (pcuAcc != NULL) cudaFree(pcuAcc);
		if (pcuIviews != NULL) cudaFree(pcuIviews);

		ostringstream oss;
		oss << "DistanceDrivenTomo::BackprojectionTomo Error: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
		throw oss.str().c_str();
	}

	if (pcuWeightedPrjs != NULL) cudaFree(pcuWeightedPrjs);
	if (pcuAcc != NULL) cudaFree(pcuAcc);
	if (pcuIviews != NULL) cudaFree(pcuIviews);
}


// C interface
// typeProjector = 0: branchless mode, need double precision
// typeProjector = 1: box integral mode, single precision only
extern "C" int cDistanceDrivenTomoProjection(
	float* prj,
	const float* img,
	const float* detCenter,
	const float* src,
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
	float du,
	float dv,
	float off_u,
	float off_v,
	int typeProjector
)
{
	float* pcuPrj = NULL;
	float* pcuImg = NULL;
	float* pcuDetCenter = NULL;
	float* pcuSrc = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nu * nv * nview * nBatches))
		{
			throw ("pcuPrj allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nx * ny * nz * nBatches))
		{
			throw ("pcuImg allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetCenter, sizeof(float3) * nview))
		{
			throw ("pcuDetCenter allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuSrc, sizeof(float3) * nview))
		{
			throw ("pcuSrc allocation failed");
		}

		cudaMemcpy(pcuImg, img, sizeof(float) * nx * ny * nz * nBatches, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuDetCenter, detCenter, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuSrc, src, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemset(pcuPrj, 0, sizeof(float) * nu * nview * nv * nBatches);

		DistanceDrivenTomo projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, du, dv, off_u, off_v, 0, 0, typeProjector
		);

		projector.ProjectionTomo(pcuImg, pcuPrj, pcuDetCenter, pcuSrc);
		cudaMemcpy(prj, pcuPrj, sizeof(float) * nu * nv * nview * nBatches, cudaMemcpyDeviceToHost);
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cDistanceDrivenTomoProjection() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	if (pcuPrj != NULL) cudaFree(pcuPrj);
	if (pcuImg != NULL) cudaFree(pcuImg);
	if (pcuDetCenter != NULL) cudaFree(pcuDetCenter);
	if (pcuSrc != NULL) cudaFree(pcuSrc);

	return cudaGetLastError();

}

extern "C" int cupyDistanceDrivenTomoProjection(
	float* prj,
	const float* img,
	const float* detCenter,
	const float* src,
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
	float du,
	float dv,
	float off_u,
	float off_v,
	int typeProjector
)
{
	try
	{
		DistanceDrivenTomo projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, du, dv, off_u, off_v, 0, 0, typeProjector
		);

		projector.ProjectionTomo(img, prj, detCenter, src);
	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cupyDistanceDrivenTomoProjection() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();
}


// C interface
extern "C" int cDistanceDrivenTomoBackprojection(
	float* img,
	const float* prj,
	const float* detCenter,
	const float* src,
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
	float du,
	float dv,
	float off_u,
	float off_v,
	int typeProjector
)
{
	float* pcuPrj = NULL;
	float* pcuImg = NULL;
	float* pcuDetCenter = NULL;
	float* pcuSrc = NULL;

	try
	{
		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nu * nv * nview * nBatches))
		{
			throw ("pcuPrj allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nx * ny * nz * nBatches))
		{
			throw ("pcuImg allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuDetCenter, sizeof(float3) * nview))
		{
			throw ("pcuDetCenter allocation failed");
		}
		if (cudaSuccess != cudaMalloc(&pcuSrc, sizeof(float3) * nview))
		{
			throw ("pcuSrc allocation failed");
		}

		cudaMemset(pcuImg, 0, sizeof(float) * nx * ny * nz * nBatches);
		cudaMemcpy(pcuDetCenter, detCenter, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuSrc, src, sizeof(float3) * nview, cudaMemcpyHostToDevice);
		cudaMemcpy(pcuPrj, prj, sizeof(float) * nu * nv * nview * nBatches, cudaMemcpyHostToDevice);

		DistanceDrivenTomo projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, du, dv, off_u, off_v, 0, 0, typeProjector
		);

		projector.BackprojectionTomo(pcuImg, pcuPrj, pcuDetCenter, pcuSrc);
		cudaMemcpy(img, pcuImg, sizeof(float) * nx * ny * nz * nBatches, cudaMemcpyDeviceToHost);

	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cDistanceDrivenTomoProjection() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	if (pcuPrj != NULL) cudaFree(pcuPrj);
	if (pcuImg != NULL) cudaFree(pcuImg);
	if (pcuDetCenter != NULL) cudaFree(pcuDetCenter);
	if (pcuSrc != NULL) cudaFree(pcuSrc);

	return cudaGetLastError();

}

extern "C" int cupyDistanceDrivenTomoBackprojection(
	float* img,
	const float* prj,
	const float* detCenter,
	const float* src,
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
	float du,
	float dv,
	float off_u,
	float off_v,
	int typeProjector
)
{
	try
	{
		DistanceDrivenTomo projector;
		projector.Setup(
			nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
			nu, nv, nview, du, dv, off_u, off_v, 0, 0, typeProjector
		);

		projector.BackprojectionTomo(img, prj, detCenter, src);

	}
	catch (exception& e)
	{
		ostringstream oss;
		oss << "cDistanceDrivenTomoProjection() failed: " << e.what()
			<< " (" << cudaGetErrorString(cudaGetLastError()) << ")";
		cerr << oss.str() << endl;
	}

	return cudaGetLastError();

}
