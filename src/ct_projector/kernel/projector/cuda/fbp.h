#pragma once

#include "projector.h"

#define FILTER_HAMMING 1
#define FILTER_HANN 2
#define FILTER_COSINE 3

class fbpFan: public Projector
{
public:
	fbpFan(): Projector() {}
	~fbpFan() {}

public:
	void Filter(float* pcuFPrj, const float* pcuPrj);
	void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) override;
};

class fbpParallel: public Projector
{
public:
	fbpParallel(): Projector() {}
	~fbpParallel() {}

public:
	void Filter(float* pcuFPrj, const float* pcuPrj);
	void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) override;
};


/* 
	FBP for helical CT based on parallel rebinned projections.

	Flohr, T.G., Bruder, H., Stierstorfer, K., Petersilka, M., Schmidt, B. and McCollough, C.H., 2008.
	Image reconstruction and image quality evaluation for a dual source CT scanner.
	Medical physics, 35(12), pp.5882-5897.
*/
class fbpHelicalParallelRebin: public Projector
{
public:
	int nviewPerPI;		// number of projections per PI segment
	float theta0; 		// start theta angle for the projections
	float zrot;			// increase in z per rotation (2PI)
	int mPI;			// the BP will look within [theta - mPI * PI, theta + mPI * PI]
	float Q;			// smoothing parameter for weighting

public:
	fbpHelicalParallelRebin();
	~fbpHelicalParallelRebin() {}

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
		int nviewPerPI,
		float theta0,
		float zrot,
		int mPI,
		float Q,
		int typeProjector = 0
	);

public:
	// pDeg is not used. Due to the PI line calculation, the angles are required to be evenly sampled.
	void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) override;
};

// FBP tomo takes the FDK framework with filter correction according to Mertelmeier et al. 2006
class fbpTomo: public Projector
{
public:
	float cutoffX;  // relative cutoff frequency compared to 2*dx
	float cutoffZ;  // relative cutoff frequency compared to 2*dz

public:
	fbpTomo(): Projector() {cutoffX = 1; cutoffZ = 1;}
	~fbpTomo() {}
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
		int typeProjector = 0,
		float cutoffX = 1,
		float cutoffZ = 1
	);

public:
	// assuming detU always at (1, 0, 0) and detV at (0, 1, 0)
	void Filter(float* pcuFPrj, const float* pcuPrj, const float* pcuDetCenter, const float* pcuSrc);

};