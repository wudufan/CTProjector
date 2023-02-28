# v0.7.2 02/27/2023

## MAJOR BUG FIX
- `tomo.distance_driven_bp()`: previous version failed to apply the ray intersection weighting during BP. The block argument for the BP kernel calling incorrectly set the last dimension as 1, which should be `nview` (line 235 in `distanceDriven3DBoxInt.cu`). The bug has been fixed.