'''
Rebinning equiangular helical to parallel conebeam and perform reconstruction.

Reference
Flohr, T.G., Bruder, H., Stierstorfer, K., Petersilka, M., Schmidt, B. and McCollough, C.H., 2008.
Image reconstruction and image quality evaluation for a dual source CT scanner.
Medical physics, 35(12), pp.5882-5897.
'''

# %%
from typing import Tuple
import numpy as np
import copy

from .ct_projector import ct_projector


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


def get_rebin_geometry(
    projector_origin: ct_projector,
    du_rebin: float = None
) -> Tuple[ct_projector, int, int]:
    '''
    Get the rebinned geometry.
    projector_origin must have the attribute rotview, number of views per rotation.

    Parameters
    --------------
    projector_origin: ct_projector.
        The geometry of the original equiangular helical beam.
    du_rebin: float.
        The pixel width after rebin. If none, it will be du / dsd * dso.

    Returns
    --------------
    projector_rebin: ct_projector.
        The geometry of the rebinned parallel helical beam.
    nview_margin_pre: int.
        The number of views needed before the target angle for rebinning.
    nview_margin_post: int.
        The number of views needed after the target angle for rebinning.
    '''

    projector_rebin = copy.deepcopy(projector_origin)

    # detector pixel size in angle
    da = projector_origin.du / projector_origin.dsd
    # the detector pixel size in mm after rebin.
    # TODO: it should be changed to half the du or passed to the function.
    if du_rebin is None:
        du_rebin = da * projector_origin.dso
    # source sampling angle interval
    dtheta = 2 * np.pi / projector_origin.rotview

    cu = (projector_origin.nu - 1) / 2 + projector_origin.off_u
    beta_min = -cu * da
    beta_max = (projector_origin.nu - 1 - cu) * da
    bmin = projector_origin.dso * np.sin(beta_min)
    bmax = projector_origin.dso * np.sin(beta_max)
    # shrink the fov a little so that there will be no gap when padding the projections along u
    cu_rebin = np.floor(-bmin / du_rebin)
    nu_rebin = int(np.floor(-bmin / du_rebin) + np.floor(bmax / du_rebin))
    off_u_rebin = cu_rebin - (nu_rebin - 1) / 2

    # number of views needed before and after for the parallel rebinning
    nview_margin_pre = int(np.ceil(beta_max / dtheta))
    nview_margin_post = int(np.ceil(-beta_min / dtheta))

    projector_rebin.du = du_rebin
    projector_rebin.nu = nu_rebin
    projector_rebin.off_u = off_u_rebin

    return projector_rebin, nview_margin_pre, nview_margin_post


def rebin_to_parallel_conebeam(
    prj: np.array,
    nview_rebin: int,
    nb: int,
    cb: float,
    db: float,
    cu: float,
    da: float,
    dso: float,
    dtheta: float,
    theta_rebin_start: float,
    theta_prj_start: float
) -> np.array:
    '''
    Kernel function to rebin helical to parallel conebeam.

    Parameters
    ---------------
    prj: np.array of shape [nview, nv, nu].
        Original projection.
    nview_rebin: int.
        Number of views after rebinning. It should remove the nview_margin_pre and
        nview_margin_post from the original number of projections.
    nb: int.
        Number of pixels per row after rebinning.
    cb: float.
        Center of row in pixels after rebinning.
    db: float.
        Width of pixel in mm after rebinning.
    cu: float.
        Center of row in pixels before rebinning.
    da: float.
        Width of pixel in radius before rebinning.
    dso: float.
        Distance between source and rotation center rebinning.
    dtheta: float.
        Sampling angle interval (same before and after rebinning).
    theta_rebin_start: float.
        The starting theta (source angle) that can be rebinned. Which is nview_margin_pre
        away from the first angle in the original projection.
    theta_prj_start: float.
        The first angle of prj.
    '''
    # row position relative to center in mm after rebin
    bs = (np.arange(nb) - cb) * db
    # corresponding fan angle for each rebinned pixel
    betas = np.arcsin(bs / dso)
    # the source angle for each view
    thetas = np.arange(nview_rebin) * dtheta + theta_rebin_start
    # before rebin dimensions
    nview = prj.shape[0]
    nv = prj.shape[1]
    nu = prj.shape[2]

    # interpolate fan angle to match the rays in the parallel beams
    print('Beta (fan angle) interpolation...')
    us = betas / da + cu
    prj_beta_rebin = np.zeros([nview, nv, nb], np.float32)
    for ib in range(nb):
        if (ib + 1) % 100 == 0:
            print(ib + 1, end=',', flush=True)

        u = us[ib]
        u0 = int(u)
        u1 = u0 + 1
        w = u - u0

        if u0 >= 0 and u0 < nu:
            prj_beta_rebin[..., ib] += (1 - w) * prj[..., u0]
        if u1 >= 0 and u1 < nu:
            prj_beta_rebin[..., ib] += w * prj[..., u1]
    print('')

    # then interpolate theta
    print('Theta (source angle) interpolation...')
    prj_rebin = np.zeros([len(thetas), nv, nb], np.float32)
    for ib in range(nb):
        if (ib + 1) % 100 == 0:
            print(ib + 1, end=',')

        # the corresponding source angle before rebinning
        alphas = (thetas - betas[ib] - theta_prj_start) / dtheta

        alphas0 = alphas.astype(int)
        alphas1 = alphas0 + 1
        w = alphas - alphas0

        valid_inds0 = np.where((alphas0 >= 0) & (alphas0 < nview))[0]
        alphas0 = alphas0[valid_inds0]
        prj_rebin[valid_inds0, :, ib] += (1 - w[valid_inds0][:, np.newaxis]) * prj_beta_rebin[alphas0, :, ib]

        valid_inds1 = np.where((alphas1 >= 0) & (alphas1 < nview))[0]
        alphas1 = alphas1[valid_inds1]
        prj_rebin[valid_inds1, :, ib] += w[valid_inds1][:, np.newaxis] * prj_beta_rebin[alphas1, :, ib]
    print('')

    return prj_rebin
