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

// FBP tomo takes the FDK framework with filter correction according to Mertelmeier et al. 2006
class fbpTomo: public Projector
{
public:
	float cutoffX;  // relative cutoff frequency compared to 2*dx
	float cutoffZ;  // relative cutoff frequency compared to 2*dz

public:
	fbpTomo(): Projector() {cutoffX = 1; cutoffZ = 1;}
	~fbpTomo() {}
	void Setup(int nBatches, 
		size_t nx, size_t ny, size_t nz, float dx, float dy, float dz, float cx, float cy, float cz,
		size_t nu, size_t nv, size_t nview, float du, float dv, float off_u, float off_v,
		float dsd, float dso, int typeProjector = 0, float cutoffX = 1, float cutoffZ = 1);

public:
	// assuming detU always at (1, 0, 0) and detV at (0, 1, 0)
	void Filter(float* pcuFPrj, const float* pcuPrj, const float* pcuDetCenter, const float* pcuSrc);

};