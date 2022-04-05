/*
The tensorflow interface for siddon cone forward and backprojection
*/

// #ifndef GOOGLE_CUDA
// #error "Only support cuda version!"
// #endif

#define EIGEN_USE_GPU

#include "projectorBase.h"

#include "siddonCone.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <sstream>

using namespace tensorflow;
using namespace std;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
using ::tensorflow::shape_inference::DimensionHandle;

/*

Siddon cone forward projection tensorflow op

Attributes:
default_shape: the default shape of the projection. when -1 is passed to any dimension of default_shape,
    use the corresponding dimension of output_shape passed on-the-fly. It is useful for shape inference. 

Inputs:
image: tensor of shape [batch, channel, nz, ny, nx].
    Python should handle the shape conversion.
geometry: tensor of shape [batch, nview * 4, 3].
    It is the concatenation of [detCenter, detU, detV, src], each with the shape of [nview, 3].
grid: tensor of shape [batch, 6].
    Each parameter is (dx, dy, dz, cx, cy, cz), all in mm
detector: tensor of length [batch, 4].
    Each parameter is (du, dv, off_u, off_v), with units (mm, mm, pixel, pixel)
output_shape: tensorflow of length [batch, 3].
    Each parameter is (nview, nv, nu). It is combined with default_shape for the final projection allocation. 
    The batch dimension is ignored for output_shape and only the first record is used. This will ensure an array can be generated in the end.

Outputs:
projection: tensor of shape [batch, channel, nview, nv, nu]. Python should handle the shape conversion.

*/
// 
REGISTER_OP("SiddonConeFP")
    .Attr("default_shape: list(int) >= 3")
    .Input("image: float")
    .Input("geometry: float")
    .Input("grid: float")
    .Input("detector: float")
    .Input("output_shape: int32")
    .Output("projection: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        // check the input rank must be 5
        ::tensorflow::shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input));

        vector<int> defaultShape;
        TF_RETURN_IF_ERROR(c->GetAttr("default_shape", &defaultShape));

        std::vector<DimensionHandle> outputDim;
        outputDim.push_back(c->Dim(c->input(0), 0)); // batch
        outputDim.push_back(c->Dim(c->input(0), 1)); // channel
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
        const cudaStream_t& stream = context->eigen_gpu_device().stream();

        // Grab the input tensors
        const Tensor& imgTensor = context->input(0);
        const Tensor& geoTensor = context->input(1);
        const float3* ptrGeo = (const float3*)geoTensor.flat<float>().data();
        int batchsize = imgTensor.dim_size(0);

        // Create an output tensor
        TensorShape prjShape;
        this->getOutputShapeWithGeo(prjShape, context);
        Tensor* prjTensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, prjShape, &prjTensor));
        int nview = prjTensor->dim_size(2);

        // memory initialization
        cudaMemsetAsync(prjTensor->flat<float>().data(), 0, sizeof(float) * prjTensor->NumElements(), stream);

        // grid and detector
        vector<Grid> grid;
        vector<Detector> det;
        this->getGrid(grid, imgTensor.shape(), context);
        this->getDetector(det, prjTensor->shape(), context);
        
        // validation
        OP_REQUIRES(
            context,
            geoTensor.dim_size(0) == batchsize && geoTensor.dim_size(1) == nview * 4 && geoTensor.dim_size(2) == 3,
            errors::InvalidArgument("geometry must have shape [batch, nview*4, 3]")
        );

        // setup projector
        SiddonCone* projector = (SiddonCone*)(this->ptrProjector);
        projector->SetCudaStream(stream);
        size_t imgBatchStride = imgTensor.dim_size(4) * imgTensor.dim_size(3) * imgTensor.dim_size(2) * imgTensor.dim_size(1);
        size_t prjBatchStride = prjTensor->dim_size(4) * prjTensor->dim_size(3) * prjTensor->dim_size(2) * prjTensor->dim_size(1);
        cudaStreamSynchronize(stream);

        // do the projection for each entry in the batch
        for (int i = 0; i < batchsize; i++)
        {
            projector->Setup(
                imgTensor.dim_size(1),
                imgTensor.dim_size(4),
                imgTensor.dim_size(3),
                imgTensor.dim_size(2),
                grid[i].dx,
                grid[i].dy,
                grid[i].dz,
                grid[i].cx,
                grid[i].cy,
                grid[i].cz,
                prjTensor->dim_size(4),
                prjTensor->dim_size(3),
                prjTensor->dim_size(2),
                det[i].du,
                det[i].dv,
                det[i].off_u,
                det[i].off_v,
                0,
                0
            );
        
            // forward projection
            const float3* geo = ptrGeo + i * nview * 4;
            projector->ProjectionArbitrary(
                imgTensor.flat<float>().data() + i * imgBatchStride,
                prjTensor->flat<float>().data() + i * prjBatchStride,
                geo,
                geo + nview,
                geo + nview * 2,
                geo + nview * 3
            );
        }
        
    }


};

REGISTER_KERNEL_BUILDER(Name("SiddonConeFP").Device(DEVICE_GPU), SiddonConeFPOp<GPUDevice>);


/*

Siddon cone backward projection tensorflow op

Attributes:
default_shape: the default shape of the image.
    When -1 is passed to any dimension of default_shape, use the corresponding dimension of output_shape passed on-the-fly.
    It is useful for shape inference. 

Inputs:
prj: tensor of shape [batch, channel, nview, nv, nu].
    Python should handle the shape conversion.
geometry: tensor of shape [batch, nview * 4, 3].
    It is the concatenation of [detCenter, detU, detV, src], each with the shape of [nview, 3].
grid: tensor of shape [batch, 6]. 
    Each parameter is (dx, dy, dz, cx, cy, cz), all in mm
detector: tensor of shape [bathc, 4]. 
    Each parameter is (du, dv, off_u, off_v), with units (mm, mm, pixel, pixel)
output_shape: tensorflow of length 3, 
    Each parameter is (nview, nv, nu). It is combined with default_shape for the final projection allocation.

Outputs:
image: tensor of shape [batch, channel, nz, ny, nx].
    Python should handle the shape conversion.

*/
// 
REGISTER_OP("SiddonConeBP")
    .Attr("default_shape: list(int) >= 3")
    .Input("projection: float")
    .Input("geometry: float")
    .Input("grid: float")
    .Input("detector: float")
    .Input("output_shape: int32")
    .Output("image: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
        // check the input rank must be 5
        ::tensorflow::shape_inference::ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input));

        vector<int> defaultShape;
        TF_RETURN_IF_ERROR(c->GetAttr("default_shape", &defaultShape));

        std::vector<DimensionHandle> outputDim;
        outputDim.push_back(c->Dim(c->input(0), 0)); // batch
        outputDim.push_back(c->Dim(c->input(0), 1)); // channel
        for (int i = 0; i < 3; i++)
        {
            outputDim.push_back(c->MakeDim(defaultShape[i]));	// nz, ny, nx
        }

        c->set_output(0, c->MakeShape(outputDim));
        return Status::OK();
    });

template <typename Device>
class SiddonConeBPOp : public ProjectorBase<Device>
{
public:
    explicit SiddonConeBPOp(OpKernelConstruction* context) : ProjectorBase<Device>(context) 
    {
        this->ptrProjector = new SiddonCone;
    }

    virtual ~SiddonConeBPOp()
	{
		if (this->ptrProjector != NULL)
		{
			delete this->ptrProjector;
		}
	}

    void Compute(OpKernelContext* context) override
    {
        const cudaStream_t& stream = context->eigen_gpu_device().stream();

        // Grab the input tensors
        const Tensor& prjTensor = context->input(0);
        const Tensor& geoTensor = context->input(1);
        const float3* ptrGeo = (const float3*)geoTensor.flat<float>().data();
        int batchsize = prjTensor.dim_size(0);
        int nview = prjTensor.dim_size(2);

        // Create an output tensor
        TensorShape imgShape;
        this->getOutputShapeTensor(imgShape, context);
        Tensor* imgTensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, imgShape, &imgTensor));

        // memory initialization
        cudaMemsetAsync(imgTensor->flat<float>().data(), 0, sizeof(float) * imgTensor->NumElements(), stream);

        // grid and detector
        vector<Grid> grid;
        vector<Detector> det;
        this->getGrid(grid, imgTensor->shape(), context);
        this->getDetector(det, prjTensor.shape(), context);
        
        // validation
        OP_REQUIRES(
            context,
            geoTensor.dim_size(0) == batchsize && geoTensor.dim_size(1) == nview * 4 && geoTensor.dim_size(2) == 3,
            errors::InvalidArgument("geometry must have shape [batch, nview*4, 3]")
        );

        // setup projector
        SiddonCone* projector = (SiddonCone*)(this->ptrProjector);
        projector->SetCudaStream(stream);
        size_t imgBatchStride = imgTensor->dim_size(4) * imgTensor->dim_size(3) * imgTensor->dim_size(2) * imgTensor->dim_size(1);
        size_t prjBatchStride = prjTensor.dim_size(4) * prjTensor.dim_size(3) * prjTensor.dim_size(2) * prjTensor.dim_size(1);
        cudaStreamSynchronize(stream);

        // backprojection
        for (int i = 0; i < batchsize; i++)
        {
            projector->Setup(
                imgTensor->dim_size(1),
                imgTensor->dim_size(4),
                imgTensor->dim_size(3),
                imgTensor->dim_size(2),
                grid[i].dx,
                grid[i].dy,
                grid[i].dz,
                grid[i].cx,
                grid[i].cy,
                grid[i].cz,
                prjTensor.dim_size(4),
                prjTensor.dim_size(3),
                prjTensor.dim_size(2),
                det[i].du,
                det[i].dv,
                det[i].off_u,
                det[i].off_v,
                0,
                0
            );
            
            const float3* geo = ptrGeo + i * nview * 4;
            projector->BackprojectionArbitrary(
                imgTensor->flat<float>().data() + i * imgBatchStride,
                prjTensor.flat<float>().data() + i * prjBatchStride,
                geo,
                geo + nview,
                geo + nview * 2,
                geo + nview * 3
            );
        }
    }


};

REGISTER_KERNEL_BUILDER(Name("SiddonConeBP").Device(DEVICE_GPU), SiddonConeBPOp<GPUDevice>);