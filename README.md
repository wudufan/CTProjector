## Projector for CT
Change the gpu-architecture in makefile under */cuda to target for different GPUs.

### Image axis
The images are assumed to have dimension (batch, nz, ny, nx), where batch is the highest dimension;

The projections are assumed to have dimension (batch, nview, nv, nu), where batch is the highest dimension.

### Notice
The cuda kernel accept float32 type array unless specified otherwise. A type cast is done within the function for C-routine python calls, but not realized for cupy-routines.

**IMPORTANT** Pay attention to the data order of arrays. Use np.copy(x, order='C') if you are not sure of the order. Sometimes if you do a slice from an array the underlying data will not be reordered. 

## Python Interface
- `projector/ct_projector.py` provides C-routine calls. It accepts numpy arrays and return numpy arrays. 
- `projector/ct_projector_cupy.py` provides cupy-routine calls. It accepts cupy arrays and return cupy arrays, so that there is no memory transfer overhead. This routine is recommended if GPU memory is enough. Cupy also provides most of the neccessary matrix operations for image reconstruction, so it could accelerate the reconstruction a lot by keeping every operation in the GPU. 

## Existing module

Beam | Detector | Trajectory | Algorithm | Comments 
---- | ---- | ---- | ---- | ----
Conebeam| Flat panel| Abitrary | Siddon | None
Conebeam| Flat panel| Tomosynthesis | Distance-driven| The main axis should always be z. Detector assumed u=(1,0,0), v=(0,1,0)
Fanbeam | Equiangular| Circular | Siddon | numpy-only
Fanbeam | Equiangular| Circular | FBP | numpy-only. Filter + pixel-driven BP.
Fanbeam | Equiangular| Circular | Distance-driven | cupy-only