#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <sstream>

#include "projector.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

// base class 
template <typename Device>
class ProjectorBase : public OpKernel
{
public:
    explicit ProjectorBase(OpKernelConstruction* context) : OpKernel(context) 
    {
        ptrProjector = NULL;

        OP_REQUIRES_OK(context, context->GetAttr("default_shape", &defaultShape));
        OP_REQUIRES(
            context,
            defaultShape.size() == 3,
            errors::InvalidArgument("default_shape must have 3 elements")
        );
    }

protected:
    void getGrid(
        std::vector<Grid>& grid,
        const TensorShape& imgShape,
        OpKernelContext* context,
        const char* input_name = "grid"
    )
    {
        int batchsize = context->input(0).dim_size(0);
        const int N = 6;

        const Tensor* ptr = NULL;
        OP_REQUIRES_OK(context, context->input(input_name, &ptr));
        OP_REQUIRES(
            context,
            ptr->dim_size(0) == batchsize && ptr->dim_size(1) == N,
            errors::InvalidArgument("grid must have shape [batch, 6]")
        );

        float* pGrid = new float [batchsize * N];
        cudaMemcpyAsync(
            pGrid,
            ptr->flat<float>().data(),
            sizeof(float) * N * batchsize,
            cudaMemcpyDeviceToHost,
            context->eigen_gpu_device().stream()
        );
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

    void getDetector(
        std::vector<Detector>& det,
        const TensorShape& prjShape,
        OpKernelContext* context,
        const char* input_name = "detector"
    )
    {
        int batchsize = context->input(0).dim_size(0);
        const int N = 4;

        const Tensor* ptr = NULL;
        OP_REQUIRES_OK(context, context->input(input_name, &ptr));
        OP_REQUIRES(
            context,
            ptr->dim_size(0) == batchsize && ptr->dim_size(1) == N,
            errors::InvalidArgument("detector must have shape [batch, 4]")
        );

        float* pDet = new float [N * batchsize];
        cudaMemcpyAsync(
            pDet,
            ptr->flat<float>().data(),
            sizeof(float) * N * batchsize,
            cudaMemcpyDeviceToHost,
            context->eigen_gpu_device().stream()
        );
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

    template <typename T> void getCPUArray(
        std::vector<T>& array,
        OpKernelContext* context,
        const char* input_name
    )
    {
        int batchsize = context->input(0).dim_size(0);

        const Tensor* ptr = NULL;
        OP_REQUIRES_OK(context, context->input(input_name, &ptr));
        OP_REQUIRES(
            context,
            ptr->dim_size(0) == batchsize,
            errors::InvalidArgument(std::string(input_name) + "must have dim_size(0) == batchsize")
        );

        T* pArray = new T [batchsize];
        cudaMemcpyAsync(
            pArray,
            ptr->flat<float>().data(),
            sizeof(T) * batchsize,
            cudaMemcpyDeviceToHost,
            context->eigen_gpu_device().stream()
        );
        cudaStreamSynchronize(context->eigen_gpu_device().stream());

        for (int i = 0; i < batchsize; i++)
        {
            array.push_back(pArray[i]);
        }

        delete [] pArray;

    }

    void getOutputShape(
        int* pShape,
        OpKernelContext* context,
        const char* input_name = "output_shape"
    )
    {
        // combine output_shape and default_shape
        const Tensor* ptr = NULL;
        OP_REQUIRES_OK(context, context->input(input_name, &ptr));
        OP_REQUIRES(
            context,
            ptr->dim_size(1) == 3,
            errors::InvalidArgument("output_shape must have shape [None, 3]")
        );

        // only use the first record
        cudaMemcpyAsync(
            &pShape[0],
            ptr->flat<int>().data(),
            sizeof(int) * 3,
            cudaMemcpyDeviceToHost,
            context->eigen_gpu_device().stream()
        );
        cudaStreamSynchronize(context->eigen_gpu_device().stream());

        // compare with defaultShape
        for (int i = 0; i < 3; i++)
        {
            if (defaultShape[i] > 0)
            {
                pShape[i] = defaultShape[i];
            }
        }
    }

    void getOutputShapeTensor(
        TensorShape& outputShape,
        OpKernelContext* context,
        const char* input_name = "output_shape"
    )
    {
        int pShape[3] = {0};
        getOutputShape(pShape, context, input_name);
        OP_REQUIRES(
            context,
            (pShape[0] > 0) && (pShape[1] > 0) && (pShape[2] > 0),
            errors::InvalidArgument("For each element, either output_shape or default_shape must be larger than 0")
        );
        
        // get the final output shape
        TensorShape finalShape;
        finalShape.AddDim(context->input(0).dim_size(0));  // batch
        finalShape.AddDim(context->input(0).dim_size(1));  // channel
        finalShape.AddDim(pShape[0]);
        finalShape.AddDim(pShape[1]);
        finalShape.AddDim(pShape[2]);

        outputShape = finalShape;
    }

    // Derive the nview from angles
    void getOutputShapeWithAngles(
        TensorShape& outputShape,
        OpKernelContext* context,
        const char* input_name_output_shape = "output_shape",
        const char* input_name_angles = "angles"
    )
    {
        int pShape[3] = {0};
        getOutputShape(pShape, context, input_name_output_shape);

        // get the nview from geometry
        const Tensor* pAngleTensor = NULL;
        OP_REQUIRES_OK(context, context->input(input_name_angles, &pAngleTensor));
        int nview =  pAngleTensor->dim_size(1);

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
        OP_REQUIRES(
            context,
            (pShape[0] > 0) && (pShape[1] > 0) && (pShape[2] > 0), 
            errors::InvalidArgument("For each element, either output_shape or default_shape must be larger than 0")
        );
        
        // get the final output shape
        TensorShape finalShape;
        finalShape.AddDim(context->input(0).dim_size(0));  // batch
        finalShape.AddDim(context->input(0).dim_size(1));  // channel
        finalShape.AddDim(pShape[0]);  // nview
        finalShape.AddDim(pShape[1]);
        finalShape.AddDim(pShape[2]);

        outputShape = finalShape;
    }

    void getOutputShapeWithGeo(
        TensorShape& outputShape,
        OpKernelContext* context,
        const char* input_name_output_shape = "output_shape",
        const char* input_name_geometry = "geometry"
    )
    {
        int pShape[3] = {0};
        getOutputShape(pShape, context, input_name_output_shape);

        // get the nview from geometry
        const Tensor* pGeoTensor = NULL;
        OP_REQUIRES_OK(context, context->input(input_name_geometry, &pGeoTensor));
        OP_REQUIRES(
            context,
            pGeoTensor->dim_size(1) % 4 == 0,
            errors::InvalidArgument("geometry.shape[1] must be nview*4")
        );
        int nview =  pGeoTensor->dim_size(1) / 4;

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
        OP_REQUIRES(
            context,
            (pShape[0] > 0) && (pShape[1] > 0) && (pShape[2] > 0), 
            errors::InvalidArgument("For each element, either output_shape or default_shape must be larger than 0")
        );
        
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
    std::vector<int> defaultShape;

};

}