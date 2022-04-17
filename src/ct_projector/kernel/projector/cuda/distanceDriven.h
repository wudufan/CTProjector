#pragma once

#include "projector.h"

class DistanceDrivenFan: public Projector
{
public:
	// memories needed
	float* pAccX;
	float* pAccY;
	float* pAccU;
	float* pWeightedPrjs;

public:
	DistanceDrivenFan(): Projector() 
	{
		this->pAccX = NULL;
		this->pAccY = NULL;
		this->pAccU = NULL;
		this->pWeightedPrjs = NULL;

	}
	~DistanceDrivenFan() 
	{ 
		this->Free(); 
	}

public:
	void Allocate(bool forward = true, bool backward = true) override;
	void AllocateExternal(float* pAccXEx, float* pAccYEx, float* pAccUEx, float* pWeightedPrjsEx);
	void Free() override;

public:
	void Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg) override;
	void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) override;
};

class DistanceDrivenParallel: public DistanceDrivenFan
{
public:
	DistanceDrivenParallel(): DistanceDrivenFan() 
	{
	}
	~DistanceDrivenParallel() 
	{
	}

public:
	void Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg) override;
	void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) override;
};

/*
Helical equiangular conebeam projection. Use box integral
*/
/*
namespace DistanceDrivenConeProjector {
	const int equiAngular = 0;
	const int equiSpace = 1;
	const int equiAngularFbp = 2;
	const int equiSpaceFbp = 3;
}

class DistanceDrivenCone: public Projector
{
public:
	// memories needed
	float* pWeightedPrjs;

public:
	DistanceDrivenCone(): Projector() 
	{
		this->pWeightedPrjs = NULL;
	}

	~DistanceDrivenCone() 
	{ 
		this->Free(); 
	}

public:
	void Allocate(bool forward = true, bool backward = true) override;
	void AllocateExternal(float* pWeightedPrjsEx);
	void Free() override;

public:
	void ProjectionHelical(const float* pcuImg, float* pcuPrj, const float* pcuDeg, const float* pcuz);
	void BackprojectionHelical(float* pcuImg, const float* pcuPrj, const float* pcuDeg, const float* pcuz);
};
*/


/* 
For the tomosynthesis, assumed that detU is x axis, detV is y axis.
The source-to-detector-center is always within 45 degrees to the z axis.
The main axis for distance driven is always z (do not need to change axis)

typeProjector = 1: use branchless mode, otherwise use box integral mode
*/

class DistanceDrivenTomo: public Projector
{
public:
	DistanceDrivenTomo(): Projector() {}
	~DistanceDrivenTomo() {}

public:
	// forward projection tomo non-tf version
	void ProjectionTomo(
		const float* pcuImg,
		float* pcuPrj,
		const float* pcuDetCenter,
		const float* pcuSrc
	);
	// backprojection tomo non-tf version
	void BackprojectionTomo(
		float* pcuImg,
		const float* pcuPrj,
		const float* pcuDetCenter,
		const float* pcuSrc
	);

public:
	// branchless version
	void ProjectionTomoBranchless(
		const float* pcuImg,
		float* pcuPrj,
		const float* pcuDetCenter,
		const float* pcuSrc,
		double* pcuAcc,
		const int* pcuIviews,
		const float3* pcuDetU,
		const float3* pcuDetV
	);
	void BackprojectionTomoBranchless(
		float* pcuImg,
		const float* pcuPrj,
		const float* pcuDetCenter,
		const float* pcuSrc,
		float* pcuWeightedPrjs,
		double* pcuAcc,
		const int* pcuIviews
	);

	// box integral version
	void ProjectionTomoBoxInt(
		const float* pcuImg,
		float* pcuPrj,
		const float* pcuDetCenter,
		const float* pcuSrc,
		const int* pcuIviews,
		const float3* pcuDetU,
		const float3* pcuDetV
	);
	void BackprojectionTomoBoxInt(
		float* pcuImg,
		const float* pcuPrj,
		const float* pcuDetCenter,
		const float* pcuSrc,
		float* pcuWeightedPrjs,
		const int* pcuIviews
	);

};
