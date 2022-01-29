#pragma once

#include "projector.h"

class SiddonFan: public Projector
{
public:
	SiddonFan(): Projector() {}
	~SiddonFan() {}

public:
	void Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg) override;
	void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) override;
};