# new version

## UPDATE

- Switch to cu112 on the main branch

## BUG FIX

- Fix the ordered subset in `example/ex_recon_gaussian_cupy.ipynb`.

# v0.8.4.cu112 04/26/2023

## UPDATE

- Build docker with tf 2.11 and cuda 11.2
- Change tf compiler option to std=c++14
- fix tensorflow op name bug

# v0.8.4 04/19/2023

## UPDATE

- Add example for gaussian penalized iterative reconstruction `ex_recon_gaussian_cupy.ipynb`.

# v0.8.0 04/12/2022

## MAJOR UPDATE

- Added numpy routine for the `recon` module; The `recon` module realized code reuse between the numpy as cupy version through the `BACKEND` variable in the `__init__.py` in the module.

# v0.7.2 02/27/2023

## MAJOR BUG FIX
- `tomo.distance_driven_bp()`: previous version failed to apply the ray intersection weighting during BP. The block argument for the BP kernel calling incorrectly set the last dimension as 1, which should be `nview` (line 235 in `distanceDriven3DBoxInt.cu`). The bug has been fixed.