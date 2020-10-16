#include "projector.h"

class DistanceDrivenFan: public Projector
{
public:
	DistanceDrivenFan(): Projector() {}
	~DistanceDrivenFan() {}

public:
	void Projection(const float* pcuImg, float* pcuPrj, const float* pcuDeg) override;
	void Backprojection(float* pcuImg, const float* pcuPrj, const float* pcuDeg) override;
};

// For the tomosynthesis, assumed that detU is x axis, detV is y axis.
// The source-to-detector-center is always within 45 degrees to the z axis.
// The main axis for distance driven is always z (do not need to change axis)

class DistanceDrivenTomo: public Projector
{
public:
	DistanceDrivenTomo(): Projector() {}
	~DistanceDrivenTomo() {}

public:
	// branchless version
	void ProjectionTomo(const float* pcuImg, float* pcuPrj, const float* pcuDetCenter, const float* pcuSrc);
	void BackprojectionTomo(float* pcuImg, const float* pcuPrj, const float* pcuDetCenter, const float* pcuSrc);


};
