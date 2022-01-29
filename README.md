# CT Projection and Reconstruction Package

## Installation
`pip install git+https://github.com/wudufan/CTProjector.git`
`pip install git+https://github.com/wudufan/CTProjector.git --no-dependencies`

The second option is more useful when you are using conda to manage the environment. 

### CUDA Version
The kernels were built with cudatoolkit 10.1.243. It should be compatible with newer version of CUDA runtime. In case it needs to be rebuilt, refer to [Build Kernel from Source](#build-kernel-from-source).

### Tensorflow Support
Tensorflow is needed only if you need the tensorflow part of the package to be functioning. Without Tensorflow, the rest part (cupy and numpy) will still be working. Because Tensorflow integration is not so frequently required, it is not included in the setup requirement.

The pre-built so files were built with tensorflow-gpu 2.4.1, it should support later version tensorflows but not likely earlier versions. 

In case you need the package to be integrated to a lower version Tensorflow, you can build it from the source, see [Build Kernel from Source](#build-kernel-from-source).

## Projector for CT
The makefile were written to target the following GPU configurations (-gencode):

- arch=compute_60,code=sm_60: P100, etc.
- arch=compute_61,code=sm_61: GTX 1080 etc.
- arch=compute_70,code=sm_70: V100 etc.
- arch=compute_75,code=sm_75: RTX2080 etc.

Change the `GPU_CODE_FLAG` in the makefile under projector/cuda and prior/cuda to change the above options.

### Build Kernel from Source
You can rebuild the kernels from directory `src/ct_projector/kernel`, in case there is compatibility issue 

- `make cuda` will build the CUDA-only version of the package
- `make all` will build both CUDA and Tensorflow version of the package.

After rebuilding, from the package root directory, run `pip install . [--no-dependencies]` to install the package.

### Image axis
The images are assumed to have dimension (batch, nz, ny, nx), where batch is the highest dimension;

The projections are assumed to have dimension (batch, nview, nv, nu), where batch is the highest dimension.

### Notice
The cuda kernel accept float32 type array unless specified otherwise. A type cast is done within the function for C-routine python calls, but not realized for cupy-routines.

**IMPORTANT** Pay attention to the data order of arrays. Use np.copy(x, order='C') if you are not sure of the order. Sometimes if you do a slice from an array the underlying data will not be reordered. 

## Python Interface
- `projector/ct_projector.py` provides C-routine calls. It accepts numpy arrays and return numpy arrays. 
- `projector/ct_projector_cupy.py` provides cupy-routine calls. It accepts cupy arrays and return cupy arrays, so that there is no memory transfer overhead. This routine is recommended if GPU memory is enough. Cupy also provides most of the neccessary matrix operations for image reconstruction, so it could accelerate the reconstruction a lot by keeping every operation in the GPU. 
- `projector/ct_projector_tensorflow.py` provides tensorflow modules. Most of the parameters can be pre-set or pass during computation. A default output shape should be passed if shape inference is needed. 

## Modules

By tf-compatible, it means that the module is programmed to: 
- Have separate memory allocation and free functions, so that all the needed GPU memory can be allocated by Tensorflow API
- Have considered cuda stream in the function calls

Name | Beam | Detector | Trajectory | Algorithm | tf-compatible | Comments 
---- | ---- | ---- | ---- | ---- | ---- | ----
siddon_cone_fp(bp)_abitrary | Conebeam| Flat panel| Abitrary | Siddon | Yes | None
distance_driven_fp(bp)_tomo | Conebeam| Flat panel| Tomosynthesis | Distance-driven | No | The main axis should always be z. Detector assumed u=(1,0,0), v=(0,1,0)
siddon_fan_fp(bp) | Fanbeam | Equiangular| Circular | Siddon | No | numpy-only
ramp_filter/fbp_fan_bp | Fanbeam | Equiangular| Circular | FBP | No | numpy-only. Filter + pixel-driven BP
distance_driven_fan_fp(bp) | Fanbeam | Equiangular| Circular | Distance-driven | Yes | cupy-only

## Tensorflow Modules

Name | Beam | Detector | Trajectory | Algorithm | Comments 
---- | ---- | ---- | ---- | ---- | ---- 
siddon_cone_fp(bp)_abitrary | Conebeam| Flat panel| Abitrary | Siddon | None

## Priors

Name | Supported denoising method | Supported priors | Supported algorithms | Comments
---- | ---- | ---- | ---- | ----
nlm | (Guided) Non-local mean | Gaussian | SQS | Needed component for Gaussian in SQS can be realized by calling nlm

## Reconstruction Algorithms

Name | Comments
---- | ----
sqs_gaussian | cupy only