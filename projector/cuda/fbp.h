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