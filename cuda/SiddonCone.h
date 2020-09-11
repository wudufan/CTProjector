#pragma once

#include "Projector.h"

class SiddonCone: public Projector
{
public:
	SiddonCone(): Projector() {}
	~SiddonCone() {}

public:
	void ProjectionAbitrary(const float* pcuImg, float* pcuPrj, const float3* pcuDetCenter,
			const float3* pcuDetU, const float3* pcuDetV, const float3* pcuSrc);
	void BackprojectionAbitrary(float* pcuImg, const float* pcuPrj, const float3* pcuDetCenter,
			const float3* pcuDetU, const float3* pcuDetV, const float3* pcuSrc);

};
