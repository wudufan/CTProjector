#pragma once

#include "projector.h"

class SiddonCone: public Projector
{
public:
	SiddonCone(): Projector() {}
	~SiddonCone() {}

public:
	void ProjectionArbitrary(
		const float* pcuImg,
		float* pcuPrj,
		const float3* pcuDetCenter,
		const float3* pcuDetU,
		const float3* pcuDetV,
		const float3* pcuSrc
	);

	void BackprojectionArbitrary(
		float* pcuImg,
		const float* pcuPrj,
		const float3* pcuDetCenter,
		const float3* pcuDetU,
		const float3* pcuDetV,
		const float3* pcuSrc
	);

};
