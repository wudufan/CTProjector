#include "fbp.h"

#include "cudaMath.h"

#include <stdexcept>
#include <sstream>
#include <iostream>


fbpHelicalParallelRebin::fbpHelicalParallelRebin(): Projector()
{
	nview = 0;
	theta0 = 0;
	zrot = 0;
	mPI = 1;
	Q = 0.5f;
}

void fbpHelicalParallelRebin::Setup(int nBatches, 
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
		int typeProjector
)
{
	Projector::Setup(
        nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
        nu, nv, nview, du, dv, off_u, off_v, dsd, dso, typeProjector);

	this->nviewPerPI = nviewPerPI;
	this->theta0 = theta0;
	this->zrot = zrot;
	this->mPI = mPI;
	this->Q = Q;
}


__global__ void bpHelicalParallelRebin(
    float* pImg,
    const float* pPrj,
    size_t nview,
    const Grid grid,
    const Detector det,
    float dsd,
    float dso,
    int nviewPerPI,
    float theta0,
    float zrot,
    int mPI,
    float Q
)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= grid.nx || iy >= grid.ny || iz >= grid.nz)
	{
		return;
	}

    // In grid, the lower left corner of the image is defined as (0, 0, 0) (for IR)
    // For FBP, the center of the pixel is defined as (0, 0, 0) so a shift of 0.5 is needed.
    register float3 pt = ImgToPhysics(make_float3(ix + 0.5f, iy + 0.5f, iz + 0.5f), grid);

	// find the current pi segment
	int iStart = int((pt.z / zrot * 2 * PI - PI / 4) * nviewPerPI / PI);

	register float val, imgVal;
	register float w, totalW;
	register float theta, cosTheta, sinTheta, u, v, q;
	imgVal = 0;
	for (int i = iStart; i < iStart + nviewPerPI; i++)
	{
		totalW = 0;
		val = 0;
		for (int k = i - mPI * nviewPerPI; k <= i + mPI * nviewPerPI; k+= nviewPerPI)
		{
			if (k < 0 || k >= nview)
			{
				continue;
			}

			theta = theta0 + k * PI / nviewPerPI;

			cosTheta = __cosf(theta);
			sinTheta = __sinf(theta);

			u = pt.x * sinTheta - pt.y * cosTheta;
//			v = 0;
			v = (pt.z - zrot * (theta - asinf(u / dso) - theta0) / 2 / PI) * dsd 
                / (sqrtf(dso * dso - u * u) - (pt.x * cosTheta + pt.y * sinTheta));

			// weighting function
			q = abs(v / det.dv / det.nv * 2.0f);
			if (q <= Q)
			{
				w = 1;
			}
			else if (q < 1)
			{
				w = __cosf((q - Q) / (1 - Q) * PI / 2);
				w *= w;
			}
			else
			{
				w = 0;
			}

			u = u / det.du + (det.nu - 1.0f) / 2.0f + det.off_u;
			v = v / det.dv + (det.nv - 1.0f) / 2.0f + det.off_v;

            val += InterpolateXY(pPrj, u, v, k, det.nu, det.nv, nview, true) * w;
			totalW += w;
		}
		imgVal += val / (totalW + 1e-6f);
//		imgVal += totalW;
	}

	pImg[iz * grid.nx * grid.ny + iy * grid.nx + ix] += imgVal;

}

void fbpHelicalParallelRebin::Backprojection(float* pcuImg, const float* pcuPrj, const float* pDeg)
{
	dim3 threads(32, 16, 1);
	dim3 blocks(ceilf(nx / (float)threads.x), ceilf(ny / (float)threads.y), nz);

	for (int ib = 0; ib < nBatches; ib++)
	{
        bpHelicalParallelRebin<<<blocks, threads, 0, m_stream>>>(
            pcuImg + ib * nx * ny * nz,
            pcuPrj + ib * nu * nv * nview,
            nview,
            MakeGrid(nx, ny, nz, dx, dy, dz, cx, cy, cz),
            MakeDetector(nu, nv, du, dv, off_u, off_v),
            dsd, dso,
            nviewPerPI, theta0, zrot, mPI, Q);
        cudaDeviceSynchronize();
	}

}

extern "C" int cfbpHelicalParallelRebinBackprojection(
    float* pImg,
    const float* pPrj,
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
    float dsd,
    float dso,
    int nviewPerPI,
    float theta0,
    float zrot,
    int mPI = 1,
    float Q = 0.5
)
{
	fbpHelicalParallelRebin projector;
	projector.Setup(
        nBatches, nx, ny, nz, dx, dy, dz, cx, cy, cz,
        nu, nv, nview, du, dv, off_u, off_v,
        dsd, dso,
        nviewPerPI, theta0, zrot, mPI, Q
    );

	float* pcuImg = NULL;
	float* pcuPrj = NULL;
	try
	{
		if (cudaSuccess != cudaMalloc(&pcuImg, sizeof(float) * nBatches * nx * ny * nz))
		{
			throw std::runtime_error("pcuImg allocation failed");
		}

		if (cudaSuccess != cudaMalloc(&pcuPrj, sizeof(float) * nBatches * nu * nv * nview))
		{
			throw std::runtime_error("pcuPrj allocation failed");
		}

		cudaMemcpy(pcuPrj, pPrj, sizeof(float) * nBatches * nu * nv * nview, cudaMemcpyHostToDevice);
		cudaMemset(pcuImg, 0, sizeof(float) * nBatches * nx * ny * nz);

        projector.Backprojection(pcuImg, pcuPrj, NULL);
        cudaMemcpy(pImg, pcuImg, sizeof(float) * nBatches * nx * ny * nz, cudaMemcpyDeviceToHost);
	}
	catch (std::exception &e)
	{
		std::ostringstream oss;
		oss << "cFBPHelicalParallelRebinBackprojection failed: " << e.what()
				<< "(" << cudaGetErrorString(cudaGetLastError()) << ")";
		std::cerr << oss.str() << std::endl;
	}

    if (pcuImg != NULL) cudaFree(pcuImg);
	if (pcuPrj != NULL) cudaFree(pcuPrj);

	return cudaGetLastError();
}