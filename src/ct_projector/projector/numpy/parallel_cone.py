'''
Rebinning helical to parallel conebeam and perform reconstruction.

Reference
Flohr, T.G., Bruder, H., Stierstorfer, K., Petersilka, M., Schmidt, B. and McCollough, C.H., 2008.
Image reconstruction and image quality evaluation for a dual source CT scanner.
Medical physics, 35(12), pp.5882-5897.
'''

# %%
from typing import Tuple
import numpy as np

# from .ct_projector import ct_projector


# %%
def convert_angles_and_calculate_pitch(
    angles: np.array, zposes: np.array
) -> Tuple[np.array, np.array]:
    '''
    Convert the angles from [0, 2*pi) to continuous incremental value.
    The input angles will fallback to 0 once it's past 2*pi, this function will
    add 2*pi to it so it keeps increasing.
    It will also calculate the pitch factor zrot, where z = zrot * angle / (2*pi) + z0.

    Parameters
    ---------------
    angles: np.array.
        Source angles in radius.
    zposes: np.array.
        The z position of the sources in mm.

    Returns
    ---------------
    angles: np.array
        The unwrapped angles.
    zrot: float
        The pitch factor.
    '''
    for i in range(1, len(angles)):
        if angles[i] < angles[i - 1]:
            angles[i:] += 2 * np.pi

    coef = np.polyfit(angles, zposes, 1)
    zrot = coef[0] * 2 * np.pi

    return angles, zrot

