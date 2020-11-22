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
#include <sstream>

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
    void getGrid(vector<Grid>& grid, const TensorShape& imgShape, OpKernelContext* context)
    {
        int batchsize = context->input(0).dim_size(0);
        const int N = 6;

        const Tensor* ptr = NULL;
        OP_REQUIRES_OK(context, context->input("grid", &ptr));
        OP_REQUIRES(context, ptr->dim_size(0) == batchsize && ptr->dim_size(1) == N, errors::InvalidArgument("grid must have shape [batch, 6]"));

        float* pGrid = new float [batchsize * N];
        cudaMemcpyAsync(pGrid, ptr->flat<float>().data(), sizeof(float) * N * batchsize, cudaMemcpyDeviceToHost, context->eigen_gpu_device().stream());
        cudaStreamSynchronize(context->eigen_gpu_device().stream());

        for (int i = 0; i < batchsize; i++)
        {
            Grid g;
            g.nx = imgShape.dim_size(4);
            g.ny = imgShape.dim_size(3);
            g.nz = imgShape.dim_size(2);
            g.dx = pGrid[i * N];
            g.dy = pGrid[i * N + 1];
            g.dz = pGrid[i * N + 2];
            g.cx = pGrid[i * N + 3];
            g.cy = pGrid[i * N + 4];
            g.cz = pGrid[i * N + 5];

            grid.push_back(g);
        }

        delete [] pGrid;
    }

    void getDetector(vector<Detector>& det, const TensorShape& prjShape, OpKernelContext* context)
    {
        int batchsize = context->input(0).dim_size(0);
        const int N = 4;

        const Tensor* ptr = NULL;
        OP_REQUIRES_OK(context, context->input("detector", &ptr));
        OP_REQUIRES(context, ptr->dim_size(0) == batchsize && ptr->dim_size(1) == N, errors::InvalidArgument("detector must have shape [batch, 4]"));

        float* pDet = new float [N * batchsize];
        cudaMemcpyAsync(pDet, ptr->flat<float>().data(), sizeof(float) * N * batchsize, cudaMemcpyDeviceToHost, context->eigen_gpu_device().stream());
        cudaStreamSynchronize(context->eigen_gpu_device().stream());
        
        for (int i = 0; i < batchsize; i++)
        {
            Detector d;
            d.nu = prjShape.dim_size(4);
            d.nv = prjShape.dim_size(3);
            d.du = pDet[i * N];
            d.dv = pDet[i * N + 1];
            d.off_u = pDet[i * N + 2];
            d.off_v = pDet[i * N + 3];

            det.push_back(d);
        }

        delete [] pDet;
    }

    void getOutputShape(TensorShape& outputShape, OpKernelContext* context)
    {
        // combine output_shape and default_shape
        const Tensor* ptr = NULL;
        OP_REQUIRES_OK(context, context->input("output_shape", &ptr));
        OP_REQUIRES(context, ptr->dim_size(1) == 3, errors::InvalidArgument("output_shape must have shape [None, 3]"));

        // only use the first record
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
        finalShape.AddDim(context->input(0).dim_size(0));  // batch
        finalShape.AddDim(context->input(0).dim_size(1));  // channel
        finalShape.AddDim(pShape[0]);
        finalShape.AddDim(pShape[1]);
        finalShape.AddDim(pShape[2]);

        outputShape = finalShape;
    }

    void getOutputShapeWithGeo(TensorShape& outputShape, OpKernelContext* context)
    {
        // combine output_shape and default_shape
        const Tensor* ptr = NULL;
        OP_REQUIRES_OK(context, context->input("output_shape", &ptr));
        OP_REQUIRES(context, ptr->dim_size(1) == 3, errors::InvalidArgument("output_shape must have shape [None, 3]"));

        // only use the first record
        int pShape[3] = {0};
        cudaMemcpyAsync(&pShape[0], ptr->flat<int>().data(), sizeof(int) * 3, cudaMemcpyDeviceToHost, context->eigen_gpu_device().stream());
        cudaStreamSynchronize(context->eigen_gpu_device().stream());

        // get the nview from geometry
        const Tensor& geoTensor = context->input(1);
        OP_REQUIRES(context, geoTensor.dim_size(1) % 4 == 0, errors::InvalidArgument("geometry.shape[1] must be nview*4"));
        int nview =  geoTensor.dim_size(1) / 4;

        // compare with defaultShape
        for (int i = 0; i < 3; i++)
        {
            if (defaultShape[i] > 0)
            {
                pShape[i] = defaultShape[i];
            }
            else if (i == 0)
            {
                pShape[i] = nview;
            }
        }
        OP_REQUIRES(context, (pShape[0] > 0) && (pShape[1] > 0) && (pShape[2] > 0), 
                    errors::InvalidArgument("For each element, either output_shape or default_shape must be larger than 0"));
        
        // get the final output shape
        TensorShape finalShape;
        finalShape.AddDim(context->input(0).dim_size(0));  // batch
        finalShape.AddDim(context->input(0).dim_size(1));  // channel
        finalShape.AddDim(pShape[0]);  // nview
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
image: tensor of shape [batch, channel, nz, ny, nx]. Python should handle the shape conversion.
geometry: tensor of shape [batch, nview * 4, 3]. It is the concatenation of [detCenter, detU, detV, src], each with the shape of [nview, 3].
grid: tensor of shape [batch, 6]. each parameter is (dx, dy, dz, cx, cy, cz), all in mm
detector: tensor of length [batch, 4]. each parameter is (du, dv, off_u, off_v), with units (mm, mm, pixel, pixel)
output_shape: tensorflow of length [batch, 3], each parameter is (nview, nv, nu). It is combined with default_shape for the final projection allocation. 
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
        OP_REQUIRES(context, geoTensor.dim_size(0) == batchsize && geoTensor.dim_size(1) == nview * 4 && geoTensor.dim_size(2) == 3,
                    errors::InvalidArgument("geometry must have shape [batch, nview*4, 3]"));

        // setup projector
        SiddonCone* projector = (SiddonCone*)(this->ptrProjector);
        projector->SetCudaStream(stream);
        size_t imgBatchStride = imgTensor.dim_size(4) * imgTensor.dim_size(3) * imgTensor.dim_size(2) * imgTensor.dim_size(1);
        size_t prjBatchStride = prjTensor->dim_size(4) * prjTensor->dim_size(3) * prjTensor->dim_size(2) * prjTensor->dim_size(1);
        cudaStreamSynchronize(stream);

        // do the projection for each entry in the batch
        for (int i = 0; i < batchsize; i++)
        {
            projector->Setup(imgTensor.dim_size(1), imgTensor.dim_size(4), imgTensor.dim_size(3), imgTensor.dim_size(2),
                         grid[i].dx, grid[i].dy, grid[i].dz, grid[i].cx, grid[i].cy, grid[i].cz, 
                         prjTensor->dim_size(4), prjTensor->dim_size(3), prjTensor->dim_size(2), det[i].du, det[i].dv, det[i].off_u, det[i].off_v, 0, 0);
        
            // forward projection
            const float3* geo = ptrGeo + i * nview * 4;
            projector->ProjectionAbitrary(imgTensor.flat<float>().data() + i * imgBatchStride, 
                                          prjTensor->flat<float>().data() + i * prjBatchStride, 
                                          geo, geo + nview, geo + nview * 2, geo + nview * 3);
        }
        
    }


};

REGISTER_KERNEL_BUILDER(Name("SiddonConeFP").Device(DEVICE_GPU), SiddonConeFPOp<GPUDevice>);


/*

Siddon cone backward projection tensorflow op

Attributes:
default_shape: the default shape of the image. when -1 is passed to any dimension of default_shape, use the corresponding dimension of output_shape passed on-the-fly.
    It is useful for shape inference. 

Inputs:
prj: tensor of shape [batch, channel, nview, nv, nu]. Python should handle the shape conversion.
geometry: tensor of shape [batch, nview * 4, 3]. It is the concatenation of [detCenter, detU, detV, src], each with the shape of [nview, 3].
grid: tensor of shape [batch, 6]. each parameter is (dx, dy, dz, cx, cy, cz), all in mm
detector: tensor of shape [bathc, 4]. each parameter is (du, dv, off_u, off_v), with units (mm, mm, pixel, pixel)
output_shape: tensorflow of length 3, each parameter is (nview, nv, nu). It is combined with default_shape for the final projection allocation.

Outputs:
image: tensor of shape [batch, channel, nz, ny, nx]. Python should handle the shape conversion.

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
        this->getOutputShape(imgShape, context);
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
        OP_REQUIRES(context, geoTensor.dim_size(0) == batchsize && geoTensor.dim_size(1) == nview * 4 && geoTensor.dim_size(2) == 3,
                    errors::InvalidArgument("geometry must have shape [batch, nview*4, 3]"));

        // setup projector
        SiddonCone* projector = (SiddonCone*)(this->ptrProjector);
        projector->SetCudaStream(stream);
        size_t imgBatchStride = imgTensor->dim_size(4) * imgTensor->dim_size(3) * imgTensor->dim_size(2) * imgTensor->dim_size(1);
        size_t prjBatchStride = prjTensor.dim_size(4) * prjTensor.dim_size(3) * prjTensor.dim_size(2) * prjTensor.dim_size(1);
        cudaStreamSynchronize(stream);

        // backprojection
        for (int i = 0; i < batchsize; i++)
        {
            projector->Setup(imgTensor->dim_size(1), imgTensor->dim_size(4), imgTensor->dim_size(3), imgTensor->dim_size(2),
                             grid[i].dx, grid[i].dy, grid[i].dz, grid[i].cx, grid[i].cy, grid[i].cz, 
                             prjTensor.dim_size(4), prjTensor.dim_size(3), prjTensor.dim_size(2), det[i].du, det[i].dv, det[i].off_u, det[i].off_v, 0, 0);
            
            const float3* geo = ptrGeo + i * nview * 4;
            projector->BackprojectionAbitrary(imgTensor->flat<float>().data() + i * imgBatchStride, 
                                              prjTensor.flat<float>().data() + i * prjBatchStride,
                                              geo, geo + nview, geo + nview * 2, geo + nview * 3);
        }
    }


};

REGISTER_KERNEL_BUILDER(Name("SiddonConeBP").Device(DEVICE_GPU), SiddonConeBPOp<GPUDevice>);