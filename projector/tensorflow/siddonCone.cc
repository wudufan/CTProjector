/*
The tensorflow interface for siddon cone forward and backprojection
*/

// #ifndef GOOGLE_CUDA
// #error "Only support cuda version!"
// #endif

#define EIGEN_USE_GPU

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#include "siddonCone.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace std;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using ::tensorflow::shape_inference::DimensionHandle;

// base class 
template <typename Device>
class ProjectorBase : public OpKernel 
{
public:
    explicit ProjectorBase(OpKernelConstruction* context) : OpKernel(context) 
    {
        ptrProjector = NULL;

        OP_REQUIRES_OK(context, context->GetAttr("default_shape", &defaultShape));
        OP_REQUIRES(context, defaultShape.size() == 3, errors::InvalidArgument("default_shape must have 3 elements"));
    }

protected:
    void getGrid(Grid& grid, const TensorShape& imgShape, OpKernelContext* context)
    {
        const Tensor* ptr = NULL;
        OP_REQUIRES_OK(context, context->input("grid", &ptr));
        OP_REQUIRES(context, ptr->NumElements() == 6, errors::InvalidArgument("grid must have 6 elements"));

        float pGrid[6];
        cudaMemcpyAsync(&pGrid, ptr->flat<float>().data(), sizeof(float) * 6, cudaMemcpyDeviceToHost, context->eigen_gpu_device().stream());
        cudaStreamSynchronize(context->eigen_gpu_device().stream());
        
        grid.nx = imgShape.dim_size(3);
        grid.ny = imgShape.dim_size(2);
        grid.nz = imgShape.dim_size(1);
        grid.dx = pGrid[0];
        grid.dy = pGrid[1];
        grid.dz = pGrid[2];
        grid.cx = pGrid[3];
        grid.cy = pGrid[4];
        grid.cz = pGrid[5];
    }

    void getDetector(Detector& det, const TensorShape& prjShape, OpKernelContext* context)
    {
        const Tensor* ptr = NULL;
        OP_REQUIRES_OK(context, context->input("detector", &ptr));
        OP_REQUIRES(context, ptr->NumElements() == 4, errors::InvalidArgument("detector must have 4 elements"));

        float pDet[4];
        cudaMemcpyAsync(&pDet, ptr->flat<float>().data(), sizeof(float) * 4, cudaMemcpyDeviceToHost, context->eigen_gpu_device().stream());
        cudaStreamSynchronize(context->eigen_gpu_device().stream());
        
        det.nu = prjShape.dim_size(3);
        det.nv = prjShape.dim_size(2);
        det.du = pDet[0];
        det.dv = pDet[1];
        det.off_u = pDet[2];
        det.off_v = pDet[3];
    }

    void getOutputShape(TensorShape& outputShape, OpKernelContext* context)
    {
        // combine output_shape and default_shape
        const Tensor* ptr = NULL;
        OP_REQUIRES_OK(context, context->input("output_shape", &ptr));
        OP_REQUIRES(context, ptr->NumElements() == 3, errors::InvalidArgument("output_shape must have 3 elements"));

        int pShape[3] = {0};
        cudaMemcpyAsync(&pShape[0], ptr->flat<int>().data(), sizeof(int) * 3, cudaMemcpyDeviceToHost, context->eigen_gpu_device().stream());
        cudaStreamSynchronize(context->eigen_gpu_device().stream());

        // compare with defaultShape
        for (int i = 0; i < 3; i++)
        {
            if (defaultShape[i] > 0)
            {
                pShape[i] = defaultShape[i];
            }
        }
        OP_REQUIRES(context, (pShape[0] > 0) && (pShape[1] > 0) && (pShape[2] > 0), 
                    errors::InvalidArgument("For each element, either output_shape or default_shape must be larger than 0"));
        
        // get the final output shape
        TensorShape finalShape;
        finalShape.AddDim(context->input(0).dim_size(0));
        finalShape.AddDim(pShape[0]);
        finalShape.AddDim(pShape[1]);
        finalShape.AddDim(pShape[2]);

        outputShape = finalShape;
    }

protected:
    Projector* ptrProjector;
    vector<int> defaultShape;

};


/*

Siddon cone forward projection tensorflow op

Attributes:
default_shape: the default shape of the projection. when -1 is passed to any dimension of default_shape, use the corresponding dimension of output_shape passed on-the-fly.
    It is useful for shape inference. 

Inputs:
img: tensor of shape [batch, nz, ny, nx]. Python should handle the shape conversion.
geometry: tensor of shape [nview * 4, 3]. It is the concatenation of [detCenter, detU, detV, src], each with the shape of [nview, 3].
grid: tensor of length 6. each parameter is (dx, dy, dz, cx, cy, cz), all in mm
detector: tensor of length 4. each parameter is (du, dv, off_u, off_v), with units (mm, mm, pixel, pixel)
output_shape: tensorflow of length 3, each parameter is (nview, nv, nu). It is combined with default_shape for the final projection allocation.

Outputs:
projection: tensor of shape [batch, nview, nv, nu]. Python should handle the shape conversion.

*/
// 
REGISTER_OP("SiddonConeFP")
    .Attr("default_shape: list(int) >= 3")
    .Input("img: float")
    .Input("geometry: float")
    .Input("grid: float")
    .Input("detector: float")
    .Input("output_shape: int32")
    .Output("projection: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        // check the input rank must be 4
        ::tensorflow::shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));

        vector<int> defaultShape;
        TF_RETURN_IF_ERROR(c->GetAttr("default_shape", &defaultShape));

        std::vector<DimensionHandle> outputDim;
        outputDim.push_back(c->Dim(c->input(0), 0)); // batch * channel
        for (int i = 0; i < 3; i++)
        {
            outputDim.push_back(c->MakeDim(defaultShape[i]));	// nview, nv, nu
        }

        c->set_output(0, c->MakeShape(outputDim));
        return Status::OK();
    });

template <typename Device>
class SiddonConeFPOp : public ProjectorBase<Device>
{
public:
    explicit SiddonConeFPOp(OpKernelConstruction* context) : ProjectorBase<Device>(context) 
    {
        this->ptrProjector = new SiddonCone;
    }

    virtual ~SiddonConeFPOp()
	{
		if (this->ptrProjector != NULL)
		{
			delete this->ptrProjector;
		}
	}

    void Compute(OpKernelContext* context) override
    {
        // Grab the input tensors
        const Tensor& imgTensor = context->input(0);
        const Tensor& geoTensor = context->input(1);

        // Create an output tensor
        TensorShape prjShape;
        this->getOutputShape(prjShape, context);
        Tensor* prjTensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, prjShape, &prjTensor));

        // grid and detector
        Grid grid;
        Detector det;
        this->getGrid(grid, imgTensor.shape(), context);
        this->getDetector(det, prjTensor->shape(), context);
        
    }


};

REGISTER_KERNEL_BUILDER(Name("SiddonConeFP").Device(DEVICE_GPU), SiddonConeFPOp<GPUDevice>);