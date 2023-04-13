# CT Projection and Reconstruction Package

## Version information
CUDA 11.0, Tensorflow-gpu 2.4.1 (pip installed)

Please check the tags for other pre-built versions.

## Installation
- `pip install git+https://github.com/wudufan/CTProjector.git`
- `pip install git+https://github.com/wudufan/CTProjector.git --no-dependencies`

The second option is more useful when you are using conda to manage the environment. 

## Usage
The package has the following python modules. Refer to the notebooks in `example/` for example scripts.

### Projectors
```python
import ct_projector.projector.cupy  # cupy
import ct_projector.projector.numpy  # numpy
import ct_projector.projector.tensorflow  # Tensorflow
```

### Priors/Denoisers
```python
import ct_projector.prior.cupy  # cupy
import ct_projector.prior.numpy  # numpy
```

### Reconstruction Algorithms
```python
import ct_projector.recon.cupy   # cupy
```

### Cupy vs. Numpy vs. Tensorflow
Cupy is the recommended routine if GPU memory is not an issue. It avoids the memory transfer between CPU and GPU and thus are a lot faster than the Numpy routine. The tensorflow routine are implemented so that the projectors can be part of a network to do auto backpropagation (unrolled network).

### Notice
- The images are assumed to have dimension (batch, nz, ny, nx), where batch is the highest dimension;
- The projections are assumed to have dimension (batch, nview, nv, nu), where batch is the highest dimension;
- The cuda kernel accept float32 type array unless specified otherwise. A type cast is done within the function for C-routine python calls, but not for cupy-routines. Hence, one needs to pay attention to their data type especially when calling the cupy subroutine;
- Pay attention to the data order of arrays. Use `np.copy(x, order='C')` or `cp.array(x, order='C')` if you are not sure of the order. Especially when you slice from an array the underlying data will not be reordered. 

## Requirements
### GPU
The supported GPU architectures are:

- arch=compute_60,code=sm_60: P100, etc.
- arch=compute_61,code=sm_61: GTX 1080 etc.
- arch=compute_70,code=sm_70: V100 etc.
- arch=compute_75,code=sm_75: RTX2080 etc.
- arch=compute_80,code=sm_80: A100 etc. (cuda >= 11.0)

Please ref to https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/ for more information.

### CUDA Version
The .so files were pre-built with certain cuda version. It should be compatible with newer version of CUDA runtime. In case it needs to be rebuilt, refer to [Build Kernel from Source](#build-kernel-from-source).

### Tensorflow Support
The .so files were built certain Tensorflow-gpu version. It is not likely to be compatible to other Tensorflow, including pip/conda-installed versions. See [Build Kernel from Source](#build-kernel-from-source) to rebuild the tensorflow module with other versions.

Tensorflow is needed only if you need the tensorflow part of the package to be functioning. Without Tensorflow, the rest part (cupy and numpy) will still be working. Because Tensorflow integration is not so frequently required, it is not included in the setup requirement.

### Cupy
cupy >= 8.0 is required to run the cupy modules. Without cupy, the numpy module will still work.

## Build Kernel from Source
The kernels can be rebuilt for different cuda and tensorflow versions.

### Using Docker
1. Modify the `docker/dockerfile` and `docker/requirements.txt` to the target cuda and tensorflow version. Note that a `devel` image is needed.
2. From `docker/` dir, run `./docker_build.sh` to build the docker image;
3. *(Only if you don't want to build tf-module) In `docker/compile.sh`, change `make all` to `make cuda`*;
4. From `docker/` dir, run `./docker_run_compile.sh` to compile the kernels;
5. From code root, run `pip install .` to install the package

### Using Conda
1. Create a new empty conda environment and activate it.
2. `conda install -c conda-forge cudatoolkit-dev=<ver>` to install target version cudatoolkit. The installation will take a while.
3. `conda install tensorflow-gpu=<ver>` to install target tensorflow version.
4. `cd src/ct_projector/kernel && make clean && make all` to build the full version, *Or `cd src/ct_projector/kernel && make clean && make cuda` to build the cuda-only version*
5. Activate your working cuda environment and run from code root `pip install .`. A minimal environment to run the examples can be found at `conda_env/create_runtime_env.sh`.

### Cuda-Only Version
Because the three modules (numpy, cupy, tensorflow) works independently, there is no need to build a cuda-only version separately. In any case if chose to build cuda-only version, tensorflow-gpu is no longer needed,

## Modules
By tf-compatible, it means that the module is programmed to: 

- Have separate memory allocation and free functions, so that all the needed GPU memory can be allocated by Tensorflow API
- Have considered cuda stream in the function calls

### Cupy and Numpy Modules

Cupy/Numpy | Module | Name | Detector | Trajectory | Algorithm | tf-compatible | Comments 
---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- 
cupy, numpy | cone | siddon_fp(bp)_arbitrary | Conebeam flat panel | Abitrary | Siddon | Yes |
cupy, numpy | tomo | distance_driven_fp(bp) | Conebeam flat panel | Tomosynthesis | Distance driven | No | The main axis should always be z. Detector assumed u=(1,0,0), v=(0,1,0)
numpy | fan_equiangular | siddon_fp(bp) | Fanbeam equiangular | Circular | Siddon | No |
cupy, numpy | fan_equiangular | ramp_filter, fbp_bp | Fanbeam equiangular | Circular | Pixel driven FBP | No |
cupy, numpy | fan_equiangular | distance_driven_fp(bp) | Fanbeam equiangular | Circular | Distance driven | Yes |
cupy, numpy | parallel | distance_driven_fp(bp) | Parallel beam | Circular | Distance driven | Yes |
numpy | parallel | pixel_driven_bp | Parallel beam | Circular | Pixel driven | No |
cupy, numpy | parallel | ramp_filter | Parallel beam | Circular | Filter | No |
numpy | helical_equiangular_parallel_rebin | helical parallel rebin/BP | Conebeam equiangular | Helical | Rebin to parallel and FBP | No | Rebin + parallel.ramp_filter + BP for reconstruction. There is also padding functions to handle the Siemens dual source CT. The parallel conebeam rebinning is more accurate than single-slice rebinning and introduce less error between the A and B reconstructions.

### Tensorflow Modules

Module | Name | Detector | Trajectory | Algorithm | Comments 
---- | ---- | ---- | ---- | ---- | ---- 
cone | SiddonFP(BP)Arbitrary | Conebeam flat panel | Arbitrary | Siddon | None
circular_2d | DistanceDriven2DFP(BP) | Parallel beam | Circular | Distance driven | see ex_fp_bp_dd_tf.ipynb
filters | ProjectionFilter | Parallel beam | Circular | RL, Hann | see ex_fp_bp_dd_tf.ipynb

### Priors

Cupy/Numpy | Name | Supported denoising method | Supported priors | Supported algorithms | Comments
---- | ---- | ---- | ---- | ---- | ----
cupy, numpy | nlm | (Guided) Non-local mean | Gaussian | SQS | Needed component for Gaussian in SQS can be realized by calling nlm

### Reconstruction Algorithms

Cupy/Numpy | Name | Comments
---- | ---- | ----
cupy, numpy | sqs_gaussian | 