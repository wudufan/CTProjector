/*
2D Distance Driven for both parallel and fan beam
*/

// #ifndef GOOGLE_CUDA
// #error "Only support cuda version!"
// #endif

#define EIGEN_USE_GPU

#include "projectorBase.h"

#include "distanceDriven.h"
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

// Use the tensorflow interface to allocate the memories
void AllocateBufferFP(
    Tensor& accXTensor,
    Tensor& accYTensor,
    int nx,
    int ny,
    int nz,
    OpKernelContext* context
) {
    TensorShape shapeAccX;
    shapeAccX.AddDim(nz);
    shapeAccX.AddDim(ny);
    shapeAccX.AddDim(nx + 1);
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, shapeAccX, &accXTensor));

    TensorShape shapeAccY;
    shapeAccY.AddDim(nz);
    shapeAccY.AddDim(ny + 1);
    shapeAccY.AddDim(nx);
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, shapeAccY, &accYTensor));
}

/*

Distance driven forward projection tensorflow op

Attributes:
default_shape: the default shape of the projection, [nview, nv, nu].
    When -1 is passed to any dimension of default_shape, use the corresponding dimension of
    output_shape passed on-the-fly. It is useful for shape inference.
type_geometry: 
    0: parallel beam
    1: equiangular fanbeam
    others: equiangular fanbeam
type_projector:
    The projector type. it will be passed to the projector constructor.
    It should be either 0 for IR projector, or 3 for FBP weighting for both FP and BP.

Inputs:
image: tensor of shape [batch, channel, nz, ny, nx].
    Python should handle the shape conversion.
angles: tensor of shape [batch, nview].
    The sampling angles
dso: tensor of shape [batch, 1].
    Source to rotation center distance. Not used for parallel geometry.
dsd: tensor of shape [batch, 1].
    Source to detector distance. Not used for parallel geometry.
grid: tensor of shape [batch, 6].
    Each parameter is (dx, dy, dz, cx, cy, cz), all in mm
detector: tensor of length [batch, 4].
    For parallel beam: Each parameter is (du, dv, off_u, off_v), with units (mm, mm, pixel, pixel);
    For fan beam: Each parameter is (da, dv, off_u, off_v), with units (rad, mm, pixel, pixel).
output_shape: tensorflow of length [batch, 3].
    Each parameter is (nview, nv, nu). It is combined with default_shape for the final projection allocation. 
    The batch dimension is ignored for output_shape and only the first record is used. This will ensure an array can be generated in the end.
    output_shape will be overriden by default_shape for any dimension that default_shape > 0.

Outputs:
projection: tensor of shape [batch, channel, nview, nv, nu]. Python should handle the shape conversion.

*/
// 
REGISTER_OP("DistanceDriven_2D_FP")
    .Attr("default_shape: list(int) >= 3")
    .Attr("type_geometry: int")
    .Attr("type_projector: int = 0")
    .Input("image: float")
    .Input("angles: float")
    .Input("dso: float")
    .Input("dsd: float")
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
class DistanceDriven2DFPOp : public ProjectorBase<Device>
{
protected:
    int typeGeometry;
    int typeProjector;

public:
    explicit DistanceDriven2DFPOp(OpKernelConstruction* context) : ProjectorBase<Device>(context) 
    {   
        OP_REQUIRES_OK(context, context->GetAttr("type_geometry", &typeGeometry));
        if (typeGeometry == 0) {
            this->ptrProjector = new DistanceDrivenParallel;
        }
        else {
            this->ptrProjector = new DistanceDrivenFan;
        }

        OP_REQUIRES_OK(context, context->GetAttr("type_projector", &typeProjector));
        OP_REQUIRES(
            context,
            typeProjector == 0 || typeProjector == 4,
            errors::InvalidArgument("type_projector must be 0 (IR) or 4 (force FBP weight)")
        );
    }

    virtual ~DistanceDriven2DFPOp()
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
        const Tensor& angleTensor = context->input(1);
        const float* pcuAngles = (const float*)angleTensor.flat<float>().data();
        int batchsize = imgTensor.dim_size(0);

        // Create an output tensor
        TensorShape prjShape;
        this->getOutputShapeWithAngles(prjShape, context);
        Tensor* prjTensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, prjShape, &prjTensor));
        int nview = prjTensor->dim_size(2);

        // memory initialization
        cudaMemsetAsync(prjTensor->flat<float>().data(), 0, sizeof(float) * prjTensor->NumElements(), stream);

        // dso and dsd
        vector<float> dso;
        vector<float> dsd;
        if (typeGeometry == 0) {
            // dummy initialization for parallel beam
            dso = vector<float>(batchsize, 500);
            dsd = vector<float>(batchsize, 1000);
        }
        else {
            this->getCPUArray(dso, context, "dso");
            this->getCPUArray(dsd, context, "dsd");
        }

        // grid and detector
        vector<Grid> grid;
        vector<Detector> det;
        this->getGrid(grid, imgTensor.shape(), context);
        this->getDetector(det, prjTensor->shape(), context);
        
        // validation
        OP_REQUIRES(
            context,
            angleTensor.dim_size(0) == batchsize && angleTensor.dim_size(1) == nview,
            errors::InvalidArgument("angles must have shape [batch, nview]")
        );

        // setup projector
        Projector* projector = this->ptrProjector;
        projector->SetCudaStream(stream);
        size_t imgBatchStride = imgTensor.dim_size(4) * imgTensor.dim_size(3) * imgTensor.dim_size(2) * imgTensor.dim_size(1);
        size_t prjBatchStride = prjTensor->dim_size(4) * prjTensor->dim_size(3) * prjTensor->dim_size(2) * prjTensor->dim_size(1);
        cudaStreamSynchronize(stream);

        // allocate buffer for forward projection
        Tensor accX;
        Tensor accY;
        AllocateBufferFP(accX, accY, imgTensor.dim_size(4), imgTensor.dim_size(3), imgTensor.dim_size(2), context);
        ((DistanceDrivenFan*)projector)->AllocateExternal(accX.flat<float>().data(), accY.flat<float>().data(), NULL, NULL);

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
                dsd[i],
                dso[i],
                this->typeProjector
            );

            // forward projection
            projector->Projection(
                imgTensor.flat<float>().data() + i * imgBatchStride,
                prjTensor->flat<float>().data() + i * prjBatchStride,
                pcuAngles + i * nview
            );
        }

        projector->Free();

    }

};

REGISTER_KERNEL_BUILDER(Name("DistanceDriven_2D_FP").Device(DEVICE_GPU), DistanceDriven2DFPOp<GPUDevice>);


// Use the tensorflow interface to allocate the memories
void AllocateBufferBP(
    Tensor& weightedPrjTensor,
    int nu,
    int nv,
    int nview,
    OpKernelContext* context
) {
    TensorShape shapeWeightedPrj;
    shapeWeightedPrj.AddDim(nview);
    shapeWeightedPrj.AddDim(nv);
    shapeWeightedPrj.AddDim(nu);
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, shapeWeightedPrj, &weightedPrjTensor));

}

/*

Distance driven backward projection tensorflow op
Note that it is the IR version of the 

Attributes:
default_shape: the default shape of the image, [nz, ny, nx].
    When -1 is passed to any dimension of default_shape, use the corresponding dimension of
    output_shape passed on-the-fly. It is useful for shape inference.
type_geometry: 
    0: parallel beam
    1: equiangular fanbeam
    others: equiangular fanbeam
type_projector:
    The projector type. it will be passed to the projector constructor.
    It should be either 0 for IR projector, or 3 for FBP weighting for both FP and BP.

Inputs:
projection: tensor of shape [batch, channel, nview, nv, nu].
    Python should handle the shape conversion.
angles: tensor of shape [batch, nview].
    The sampling angles
dso: tensor of shape [batch, 1].
    Source to rotation center distance. Not used for parallel geometry.
dsd: tensor of shape [batch, 1].
    Source to detector distance. Not used for parallel geometry.
grid: tensor of shape [batch, 6].
    Each parameter is (dx, dy, dz, cx, cy, cz), all in mm
detector: tensor of length [batch, 4].
    For parallel beam: Each parameter is (du, dv, off_u, off_v), with units (mm, mm, pixel, pixel);
    For fan beam: Each parameter is (da, dv, off_u, off_v), with units (rad, mm, pixel, pixel).
output_shape: tensorflow of length [batch, 3].
    Each parameter is (nz, ny, nx). It is combined with default_shape for the final projection allocation. 
    The batch dimension is ignored for output_shape and only the first record is used. This will ensure an array can be generated in the end.
    output_shape will be overriden by default_shape for any dimension that default_shape > 0.

Outputs:
image: tensor of shape [batch, channel, nz, ny, nx]. Python should handle the shape conversion.

*/
// 
REGISTER_OP("DistanceDriven_2D_BP")
    .Attr("default_shape: list(int) >= 3")
    .Attr("type_geometry: int")
    .Attr("type_projector: int = 0")
    .Input("projection: float")
    .Input("angles: float")
    .Input("dso: float")
    .Input("dsd: float")
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
class DistanceDriven2DBPOp : public ProjectorBase<Device>
{
protected:
    int typeGeometry;
    int typeProjector;

public:
    explicit DistanceDriven2DBPOp(OpKernelConstruction* context) : ProjectorBase<Device>(context) 
    {   
        OP_REQUIRES_OK(context, context->GetAttr("type_geometry", &typeGeometry));
        if (typeGeometry == 0) {
            this->ptrProjector = new DistanceDrivenParallel;
        }
        else {
            this->ptrProjector = new DistanceDrivenFan;
        }

        OP_REQUIRES_OK(context, context->GetAttr("type_projector", &typeProjector));
        OP_REQUIRES(
            context,
            typeProjector == 0 || typeProjector == 4,
            errors::InvalidArgument("type_projector must be 0 (IR) or 4 (force FBP weight)")
        );
    }

    virtual ~DistanceDriven2DBPOp()
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
        const Tensor& angleTensor = context->input(1);
        const float* pcuAngles = (const float*)angleTensor.flat<float>().data();
        int batchsize = prjTensor.dim_size(0);

        // Create an output tensor
        TensorShape imgShape;
        this->getOutputShapeWithAngles(imgShape, context);
        Tensor* imgTensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, imgShape, &imgTensor));
        int nview = prjTensor.dim_size(2);

        // memory initialization
        cudaMemsetAsync(imgTensor->flat<float>().data(), 0, sizeof(float) * imgTensor->NumElements(), stream);

        // dso and dsd
        vector<float> dso;
        vector<float> dsd;
        if (typeGeometry == 0) {
            // dummy initialization for parallel beam
            dso = vector<float>(batchsize, 500);
            dsd = vector<float>(batchsize, 1000);
        }
        else {
            this->getCPUArray(dso, context, "dso");
            this->getCPUArray(dsd, context, "dsd");
        }

        // grid and detector
        vector<Grid> grid;
        vector<Detector> det;
        this->getGrid(grid, imgTensor->shape(), context);
        this->getDetector(det, prjTensor.shape(), context);
        
        // validation
        OP_REQUIRES(
            context,
            angleTensor.dim_size(0) == batchsize && angleTensor.dim_size(1) == nview,
            errors::InvalidArgument("angles must have shape [batch, nview]")
        );

        // setup projector
        Projector* projector = this->ptrProjector;
        projector->SetCudaStream(stream);
        size_t imgBatchStride = imgTensor->dim_size(4) * imgTensor->dim_size(3) * imgTensor->dim_size(2) * imgTensor->dim_size(1);
        size_t prjBatchStride = prjTensor.dim_size(4) * prjTensor.dim_size(3) * prjTensor.dim_size(2) * prjTensor.dim_size(1);
        cudaStreamSynchronize(stream);

        // allocate buffer for forward projection
        Tensor weightedPrjs;
        AllocateBufferBP(weightedPrjs, prjTensor.dim_size(4), prjTensor.dim_size(3), prjTensor.dim_size(2), context);
        ((DistanceDrivenFan*)projector)->AllocateExternal(NULL, NULL, NULL, weightedPrjs.flat<float>().data());

        // do the projection for each entry in the batch
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
                dsd[i],
                dso[i],
                this->typeProjector
            );

            // backprojection
            projector->Backprojection(
                imgTensor->flat<float>().data() + i * imgBatchStride,
                prjTensor.flat<float>().data() + i * prjBatchStride,
                pcuAngles + i * nview
            );
        }

        projector->Free();

    }

};

REGISTER_KERNEL_BUILDER(Name("DistanceDriven_2D_BP").Device(DEVICE_GPU), DistanceDriven2DBPOp<GPUDevice>);
