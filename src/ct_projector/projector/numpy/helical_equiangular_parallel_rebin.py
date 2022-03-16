'''
Rebinning equiangular helical to parallel conebeam and perform reconstruction.

Reference
Flohr, T.G., Bruder, H., Stierstorfer, K., Petersilka, M., Schmidt, B. and McCollough, C.H., 2008.
Image reconstruction and image quality evaluation for a dual source CT scanner.
Medical physics, 35(12), pp.5882-5897.
'''

# %%
from ctypes import cdll, POINTER, c_float, c_int, c_ulong
from typing import Tuple
import numpy as np
import copy

from .ct_projector import ct_projector
from .parallel import ramp_filter

import pkg_resources

module = cdll.LoadLibrary(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libprojector.so')
)


# %%
def _convert_angles_and_calculate_pitch(
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


def _get_rebin_geometry(
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


def _rebin_to_parallel_conebeam(
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

    # transpose the prj to nu, nview, nv for faster memory access
    prj = np.copy(prj.transpose(2, 0, 1), 'C')

    # before rebin dimensions
    nview = prj.shape[1]
    nv = prj.shape[2]
    nu = prj.shape[0]

    # interpolate fan angle to match the rays in the parallel beams
    print('Beta (fan angle) interpolation...')
    us = betas / da + cu
    prj_beta_rebin = np.zeros([nb, nview, nv], np.float32)
    for ib in range(nb):
        if (ib + 1) % 100 == 0:
            print(ib + 1, end=',', flush=True)

        u = us[ib]
        u0 = int(u)
        u1 = u0 + 1
        w = u - u0

        if u0 >= 0 and u0 < nu:
            prj_beta_rebin[ib, ...] += (1 - w) * prj[u0, ...]
        if u1 >= 0 and u1 < nu:
            prj_beta_rebin[ib, ...] += w * prj[u1, ...]
    print('')

    # then interpolate theta
    print('Theta (source angle) interpolation...')
    prj_rebin = np.zeros([nb, len(thetas), nv], np.float32)
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
        prj_rebin[ib, valid_inds0, :] += (1 - w[valid_inds0][:, np.newaxis]) * prj_beta_rebin[ib, alphas0, :]

        valid_inds1 = np.where((alphas1 >= 0) & (alphas1 < nview))[0]
        alphas1 = alphas1[valid_inds1]
        prj_rebin[ib, valid_inds1, :] += w[valid_inds1][:, np.newaxis] * prj_beta_rebin[ib, alphas1, :]
    print('')

    # transpose back
    prj_rebin = np.copy(prj_rebin.transpose((1, 2, 0)), 'C')

    return prj_rebin


def rebin_helical_to_parallel(
    projector: ct_projector,
    prjs: np.array,
    angles: np.array,
    zposes: np.array
) -> Tuple[ct_projector, np.array, np.array, float, int, int]:
    '''
    Rebin helical data to parallel geometry.

    Parameters
    -------------------
    projector: ct_projector.
        Geometry of the helical beam.
    prjs: np.array of shape [nview, nv, nu]
        Projection to be rebinned.
    angles: np.array of shape [nview]
        The source angle for each view.
    zposes: np.array of shape [nview]
        The z pos of source for each view.

    Returns
    --------------------
    projector_rebin: ct_projector.
        The rebinned geometry.
    prjs_rebin: np.array of shape [nview_rebin, nv, nu_rebin].
        The rebinned projection.
    angles: np.array of shape [nview].
        The angles unwrapped from [0, 2*pi).
    zrot: float.
        The source displacement in z per rotation: z = zrot * angle / (2*pi) + z0.
    nview_margin_pre: int.
        The offset from the first projection to the first rebinned view.
    nview_margin_post: int.
        The offset from the last rebinned view to the last projection.
    '''
    dtheta = 2 * np.pi / projector.rotview
    angles, zrot = _convert_angles_and_calculate_pitch(angles, zposes)
    projector_rebin, nview_margin_pre, nview_margin_post = _get_rebin_geometry(projector)
    prjs_rebin = _rebin_to_parallel_conebeam(
        prjs,
        prjs.shape[0] - nview_margin_pre - nview_margin_post,
        projector_rebin.nu,
        (projector_rebin.nu - 1) / 2 + projector_rebin.off_u,
        projector_rebin.du,
        (projector.nu - 1) / 2 + projector.off_u,
        projector.du / projector.dsd,
        projector.dso,
        dtheta,
        angles[nview_margin_pre],
        angles[0]
    )

    return projector_rebin, prjs_rebin, angles, zrot, nview_margin_pre, nview_margin_post


# %%
# For Siemens dual source CT geometry
class PaddingParams:
    '''
    The parameters for dect padding.

    Parameters
    ---------------
    first_angle_offset: int
        The projection B acquired at angles_b[0] is corresponding to the projection A
        acquired at angles_b[first_angle_offset].
    istart_rebin_pad_a: int
        Projection A start for padding relative to rebinned projection.
    istart_rebin_pad_b: int
        Projection B start for padding relative to rebinned projection.
    iend_rebin_pad_a: int
        Projection A end for padding relative to rebinned projection.
    iend_rebin_pad_b: int
        Projection B end for padding relative to rebinned projection.
    iprj_offset_same: int
        Projection offset A to B for padding along the same direction.
    iprj_offset_opp: int
        Projection offset A to B for padding along the opposite direction.
    z_offset_same: float
        Z offset A to B for padding along the same direction.
    z_offset_opp: float
        Z offset A to B for padding along the opposite direction.
    '''
    def __init__(
        self,
        first_angle_offset,
        istart_rebin_pad_a,
        istart_rebin_pad_b,
        iend_rebin_pad_a,
        iend_rebin_pad_b,
        iprj_offset_same,
        iprj_offset_opp,
        z_offset_same,
        z_offset_opp
    ):
        self.first_angle_offset = first_angle_offset
        self.istart_rebin_pad_a = istart_rebin_pad_a
        self.istart_rebin_pad_b = istart_rebin_pad_b
        self.iend_rebin_pad_a = iend_rebin_pad_a
        self.iend_rebin_pad_b = iend_rebin_pad_b
        self.iprj_offset_same = iprj_offset_same
        self.iprj_offset_opp = iprj_offset_opp
        self.z_offset_same = z_offset_same
        self.z_offset_opp = z_offset_opp


def _get_dect_padding_parameters(
    angles_a: np.array,
    angles_b: np.array,
    first_angle_offset: int,
    nview_margin_pre_a: int,
    nview_margin_pre_b: int,
    nview_margin_post_a: int,
    nview_margin_post_b: int,
    dtheta: float,
    zrot: float,
    rotview: int
) -> PaddingParams:
    '''
    Calculate the padding related parameters for Siemens dual-source CT

    Parameters
    ----------------
    angles_a: np.array of shape [nview_a].
        The source angles of detector A (wider detector) in radius.
    angles_b: np.array of shape [nview_b].
        The source angles of detector B (narrower detector) in radius.
    first_angle_offset: int.
        The projection B acquired at angles_b[0] is corresponding to the projection A
        acquired at angles_b[first_angle_offset].
    nview_margin_pre_a: int.
        Number of views from angles_a[0] to first angle in rebinned projection A.
    nview_margin_pre_b: int.
        Number of views from angles_b[0] to first angle in rebinned projection B.
    nview_margin_post_a: int.
        Number of views from the last angle in rebinned projection A to angles_a[-1].
    nview_margin_post_b: int.
        Number of views from the last angle in rebinned projection B to angles_b[-1].
    dtheta: float.
        The angular sampling interval.
    zrot: float.
        z step every rotation: z = zrot * angle / (2 * np.pi) + z0.
    rotview: int.
        Number of views per rotation.

    Returns
    ------------------
    padding_params: PaddingParams
        The paremeters for dect padding.
    '''
    # the angle from A to B acquired at the same time point.
    angle_offset = angles_b[0] - angles_a[first_angle_offset]
    # wrap between [-pi, pi)
    angle_offset = angle_offset - int(angle_offset / np.pi) * np.pi
    # projection A needs to shift this value for same direction padding to projection B
    iprj_offset_same = int(angle_offset / dtheta)
    # z_offset_same = zrot * iprj_offset_same * dtheta / (2 * np.pi)
    z_offset_same = zrot * angle_offset / (2 * np.pi)

    # projection A needs to shift this value for opposite direction padding
    # It should go to a different direction than the same padding offset
    if iprj_offset_same > 0:
        iprj_offset_opp = iprj_offset_same - int(rotview / 2)
        z_offset_opp = z_offset_same - zrot / 2
    else:
        iprj_offset_opp = iprj_offset_same + int(rotview / 2)
        z_offset_opp = z_offset_same + zrot / 2

    # The projection A shift needed to pad the first and last projection B
    # They are used if projection B is needed to be truncated if there is not much margin
    # between projection B and projection A.
    iprj_offset_first = min(iprj_offset_same, iprj_offset_opp)
    iprj_offset_last = max(iprj_offset_same, iprj_offset_opp)

    # get the starting position
    # Rebinned projection A and B starting position relative to angles_a[0]
    istart_origin_rebin_a = nview_margin_pre_a
    istart_origin_rebin_b = first_angle_offset + nview_margin_pre_b
    # The first A projection to pad B, relative to angles_a[0]
    istart_origin_pad_a = istart_origin_rebin_b + iprj_offset_first
    # Calculate the A and B projections can be used for padding relative to the rebinned starting
    if istart_origin_pad_a < istart_origin_rebin_a:
        # if A for padding is not valid in the rebinned projections
        # push both A and B start so A falls into the rebinned range
        istart_rebin_pad_a = 0
        istart_rebin_pad_b = istart_origin_rebin_a - istart_origin_pad_a
    else:
        istart_rebin_pad_a = istart_origin_pad_a
        istart_rebin_pad_b = 0

    # get the ending position
    # rebinned projection A and B ending position relative to angles_a[0]
    iend_origin_rebin_a = len(angles_a) - nview_margin_post_a
    iend_origin_rebin_b = first_angle_offset + len(angles_b) - nview_margin_post_b
    # The last A projection to pad B, relative to angles_a[0]
    iend_origin_pad_a = iend_origin_rebin_b + iprj_offset_last
    # Calculate the A and B projection endings used for padding relative to the rebinned starting
    if iend_origin_pad_a > iend_origin_rebin_a:
        # if A for padding is not within the rebinned range
        # push both A and B ending so A falls into the rebinned range
        iend_rebin_pad_a = iend_origin_rebin_a - istart_origin_rebin_a
        iend_rebin_pad_b = iend_origin_rebin_b - (iend_origin_pad_a - iend_origin_rebin_a) - istart_origin_rebin_b
    else:
        iend_rebin_pad_a = iend_origin_pad_a - istart_origin_rebin_a
        iend_rebin_pad_b = iend_origin_rebin_b - istart_origin_rebin_b

    return PaddingParams(
        first_angle_offset,
        istart_rebin_pad_a,
        istart_rebin_pad_b,
        iend_rebin_pad_a,
        iend_rebin_pad_b,
        iprj_offset_same,
        iprj_offset_opp,
        z_offset_same,
        z_offset_opp,
    )


def _pad_dect_rebinned_prjs(
    prj_rebin_a: np.array,
    prj_rebin_b: np.array,
    padding_params: PaddingParams,
    cua: float,
    cub: float,
    dv: float,
    transit_len: int = 20
) -> np.array:
    '''
    Pad the rebinned projections from DECT

    Parameters
    -------------------
    prj_rebin_a: np.array of shape [nview, nv, nu].
        The rebinned projection A (wider detector)
    prj_rebin_b: np.array of shape [nview, nv, nu].
        The rebinned projection B (narrower detector)
    padding_params: PaddingParams.
        The padding parameters.
    cua: float.
        Center of rebinned detector A in pixel.
    cub: float
        Center of rebinned detector B in pixel.
    transit_len: int
        The length of smoothing area when padding projection B with A.

    Returns
    -------------------
    prj_rebin_ab: np.array of shape [2, nview, nv, nu].
        The rebinned projections A and B, with B padded to the width of A.
    '''
    # transition smoothing factor
    w = np.cos(np.pi / 2 * np.arange(transit_len) / transit_len)
    w = w * w
    w = w[np.newaxis, np.newaxis, :]

    # the area on B to be padded with A
    prj_rebin_b_ex = np.zeros(
        [
            padding_params.iend_rebin_pad_b - padding_params.istart_rebin_pad_b,
            prj_rebin_b.shape[1],
            prj_rebin_a.shape[2]
        ],
        np.float32
    )

    istart_a = padding_params.istart_rebin_pad_a + padding_params.first_angle_offset
    istart_a_opp = istart_a + padding_params.iprj_offset_opp
    istart_a_same = istart_a + padding_params.iprj_offset_same
    # padding different direction
    if padding_params.z_offset_opp < 0:
        iv = int(-padding_params.z_offset_opp / dv)
        prj_rebin_b_ex[:, :-iv, :] = prj_rebin_a[
            istart_a_opp:istart_a_opp + prj_rebin_b_ex.shape[0], iv:, ::-1,
        ]
    else:
        iv = int(padding_params.z_offset_opp / dv)
        prj_rebin_b_ex[:, iv:, :] = prj_rebin_a[
            istart_a_opp:istart_a_opp + prj_rebin_b_ex.shape[0], :-iv, ::-1,
        ]

    # padding same direction
    if padding_params.z_offset_same < 0:
        iv = int(-padding_params.z_offset_same / dv)
        prj_rebin_b_ex[:, :-iv, :] = prj_rebin_a[
            istart_a_same:istart_a_same + prj_rebin_b_ex.shape[0], iv:, :
        ]
    else:
        iv = int(padding_params.z_offset_same / dv)
        prj_rebin_b_ex[:, iv:, :] = prj_rebin_a[
            istart_a_same:istart_a_same + prj_rebin_b_ex.shape[0], :-iv, :
        ]

    # put prj_rebin_b in the middle
    offset = int(cua - cub)

    istart_b = padding_params.istart_rebin_pad_b
    iend_b = padding_params.iend_rebin_pad_b
    prj_rebin_b_ex[..., offset:offset + transit_len] = \
        prj_rebin_b[istart_b:iend_b, :, :transit_len] * (1 - w) \
        + w * prj_rebin_b_ex[..., offset:offset + transit_len]

    prj_rebin_b_ex[..., offset + prj_rebin_b.shape[2] - transit_len:offset + prj_rebin_b.shape[2]] = \
        prj_rebin_b[istart_b:iend_b, :, -transit_len:] * w \
        + (1 - w) * prj_rebin_b_ex[..., offset + prj_rebin_b.shape[2] - transit_len:offset + prj_rebin_b.shape[2]]

    prj_rebin_b_ex[..., offset + transit_len:offset + prj_rebin_b.shape[2] - transit_len] = \
        prj_rebin_b[istart_b:iend_b, :, transit_len:-transit_len]

    # truncate to same length for projection A and padded projection B
    prj_rebin_ab = np.array([
        prj_rebin_a[istart_a:istart_a + prj_rebin_b_ex.shape[0], ...],
        prj_rebin_b_ex
    ])

    return prj_rebin_ab


def _get_first_angle_ind(
    padding_params: PaddingParams,
    nview_margin_pre_a: int,
    nview_margin_pre_b: int
) -> Tuple[int, int]:
    '''
    Get the index for the first angle and zpos in the padded projections A and B.
    '''
    istart_angle_a = nview_margin_pre_a + padding_params.istart_rebin_pad_a + padding_params.first_angle_offset
    istart_angle_b = nview_margin_pre_b + padding_params.istart_rebin_pad_b

    return istart_angle_a, istart_angle_b


def pad_dual_source_ct_rebinned_projection(
    projector_rebin_a: ct_projector,
    projector_rebin_b: ct_projector,
    prjs_rebin_a: np.array,
    prjs_rebin_b: np.array,
    angles_a: np.array,
    angles_b: np.array,
    zrot_a: float,
    zrot_b: float,
    first_angle_offset: int,
    nview_margin_pre_a: int,
    nview_margin_pre_b: int,
    nview_margin_post_a: int,
    nview_margin_post_b: int,
    transit_len: int = 20
) -> Tuple[np.array, int, int]:
    '''
    Pad the rebinned projection B by projection A.

    Parameters
    -------------------
    projector_rebin_a: ct_projector
        The geometry of rebinned projection A.
    projector_rebin_b: ct_projector
        The geometry of rebinned projection B.
    prj_rebin_a: np.array of shape [nview, nv, nu].
        The rebinned projection A (wider detector)
    prj_rebin_b: np.array of shape [nview, nv, nu].
        The rebinned projection B (narrower detector)
    angles_a: np.array of shape [nview_a].
        The source angles of detector A (wider detector) in radius.
    angles_b: np.array of shape [nview_b].
        The source angles of detector B (narrower detector) in radius.
    zrot_a: float.
        The source movement in z per rotation calculated from projection A.
    zrot_b: float.
        The source movement in z per rotation calculated from projection B.
    first_angle_offset: int.
        The projection B acquired at angles_b[0] is corresponding to the projection A
        acquired at angles_b[first_angle_offset].
    nview_margin_pre_a: int.
        Number of views from angles_a[0] to first angle in rebinned projection A.
    nview_margin_pre_b: int.
        Number of views from angles_b[0] to first angle in rebinned projection B.
    nview_margin_post_a: int.
        Number of views from the last angle in rebinned projection A to angles_a[-1].
    nview_margin_post_b: int.
        Number of views from the last angle in rebinned projection B to angles_b[-1].
    transit_len: int
        The length of smoothing area when padding projection B with A.

    Returns
    -----------------
    prj_rebin_ab: np.array of shape [2, nview, nv, nu].
        The rebinned projections A and B, with B padded to the width of A.
    istart_a: int.
        The starting index in angles_a and zpos_a of the first padded projection A.
    istart_B: int.
        The starting index in angles_b and zpos_b of the first padded projection A.
    '''
    assert(projector_rebin_a.rotview == projector_rebin_b.rotview)

    padding_params = _get_dect_padding_parameters(
        angles_a,
        angles_b,
        first_angle_offset,
        nview_margin_pre_a,
        nview_margin_pre_b,
        nview_margin_post_a,
        nview_margin_post_b,
        2 * np.pi / projector_rebin_a.rotview,
        (zrot_a + zrot_b) / 2,
        projector_rebin_a.rotview
    )

    prjs_rebin_ab = _pad_dect_rebinned_prjs(
        prjs_rebin_a,
        prjs_rebin_b,
        padding_params,
        (projector_rebin_a.nu + 1) / 2 + projector_rebin_a.off_u,
        (projector_rebin_b.nu + 1) / 2 + projector_rebin_a.off_u,
        projector_rebin_a.dv,
        transit_len
    )

    istart_a, istart_b = _get_first_angle_ind(padding_params, nview_margin_pre_a, nview_margin_pre_b)

    return prjs_rebin_ab, istart_a, istart_b


# %%
'''
backprojection functions. The parallel filtering is from parallel.ramp_filter
'''


def fbp_bp(
    projector: ct_projector,
    fprj: np.array,
    theta0: float,
    zrot: float,
    m_pi: int = 1,
    q_value: float = 0.5
) -> np.array:
    '''
    Helical rebinned parallel beam backprojection, pixel driven.
    The sampling angles need to be evenly distributed.

    Parameters
    ----------------
    prj: np.array(float32) of size [batch, nview, nv, nu].
        The projection to be backprojected. The size does not need to be the same
        with projector.nview, projector.nv, projector.nu.
    theta0: float.
        The starting sampling angle.
    zrot: float.
        Source z movement per rotation.
    m_pi: int.
        The BP will look within [theta - mPI * PI, theta + mPI * PI]
    q_value: float.
        Smoothing parameter for weighting

    Returns
    --------------
    img: np.array(float32) of size [batch, projector.nz, projector.ny, projector.nx]
        The backprojected image.
    '''

    fprj = fprj.astype(np.float32)
    img = np.zeros([fprj.shape[0], projector.nz, projector.ny, projector.nx], np.float32)

    module.cfbpHelicalParallelRebinBackprojection.restype = c_int

    err = module.cfbpHelicalParallelRebinBackprojection(
        img.ctypes.data_as(POINTER(c_float)),
        fprj.ctypes.data_as(POINTER(c_float)),
        c_ulong(img.shape[0]),
        c_ulong(img.shape[3]),
        c_ulong(img.shape[2]),
        c_ulong(img.shape[1]),
        c_float(projector.dx),
        c_float(projector.dy),
        c_float(projector.dz),
        c_float(projector.cx),
        c_float(projector.cy),
        c_float(projector.cz),
        c_ulong(fprj.shape[3]),
        c_ulong(fprj.shape[2]),
        c_ulong(fprj.shape[1]),
        c_float(projector.du),
        c_float(projector.dv),
        c_float(projector.off_u),
        c_float(projector.off_v),
        c_float(projector.dsd),
        c_float(projector.dso),
        c_int(int(projector.rotview / 2)),
        c_float(theta0),
        c_float(zrot),
        c_int(m_pi),
        c_float(q_value)
    )

    if err != 0:
        print(err)

    return img * 2 / m_pi


def fbp_long(
    projector: ct_projector,
    prj: np.array,
    theta0: float,
    z0: float,
    zrot: float,
    filter_type: str = 'hann',
    recon_z_margin: float = 0,
    recon_z_start: float = None,
    recon_z_end: float = None,
    recon_z_batch_size: int = 200,
    m_pi: int = 1,
    q_value: float = 0.5
) -> np.array:
    '''
    Helical rebinned parallel beam backprojection, pixel driven.
    This function will reconstruct the whole volume segments by segments to save GPU memory.
    The sampling angles need to be evenly distributed.

    Parameters
    ----------------
    prj: np.array(float32) of size [batch, nview, nv, nu].
        The projection to be filtered and backprojected. The size does not need to be the same
        with projector.nview, projector.nv, projector.nu.
    theta0: float.
        The starting sampling angle.
    z0: float.
        The z position of the first rebinned projection.
    zrot: float.
        Source z movement per rotation.
    recon_z_start: float.
        Starting z coordinate for reconstruction. If None, will use z0 + recon_z_margin.
    recon_z_end: float.
        Ending z coordinate for reconstruction. If None, will use max projection z - recon_z_margin.
    recon_z_batch_size: int.
        Number of z slices to be included in one batch reconstruction.
    m_pi: int.
        The BP will look within [theta - mPI * PI, theta + mPI * PI]
    q_value: float.
        Smoothing parameter for weighting

    Returns
    --------------
    img: np.array(float32) of size [batch, projector.nz, projector.ny, projector.nx]
        The backprojected image.
    '''

    if recon_z_start is None:
        recon_z_start = z0 + recon_z_margin
    if recon_z_end is None:
        recon_z_end = z0 + zrot * prj.shape[1] / projector.rotview - recon_z_margin

    dtheta = 2 * np.pi / projector.rotview
    nz = int(np.ceil((recon_z_end - recon_z_start) / projector.dz))
    assert(nz > 0)

    if recon_z_batch_size > nz:
        recon_z_batch_size = nz

    img = []
    for iz in range(0, nz, recon_z_batch_size):
        # z size of the current batch
        nz_batch = min(recon_z_batch_size, nz - iz)
        # The starting and ending of the current volume in z
        z0_batch = recon_z_start + iz * projector.dz
        z1_batch = recon_z_start + (iz + nz_batch) * projector.dz

        # The projections needed for the current batch
        # starting projection
        prj_z0_batch = z0_batch - (m_pi + 0.5) * zrot / 2
        iprj_start_batch = max(int((prj_z0_batch - z0) / zrot * projector.rotview), 0)
        theta0_batch = theta0 + iprj_start_batch * dtheta
        # ending projection
        prj_z1_batch = z1_batch + (m_pi + 0.5) * zrot / 2
        iprj_end_batch = min(int(np.ceil((prj_z1_batch - z0) / zrot * projector.rotview)), prj.shape[1])

        # This is the recon volume z start relative to the current projection segment
        recon_z0_relative = z0_batch - (z0 + iprj_start_batch * zrot / projector.rotview)
        projector_batch = copy.deepcopy(projector)
        projector_batch.nz = nz_batch
        projector_batch.cz = recon_z0_relative + projector_batch.nz * projector_batch.dz / 2

        fprj_batch = ramp_filter(
            projector_batch,
            np.copy(prj[:, iprj_start_batch:iprj_end_batch, :, :], 'C'),
            filter_type
        )
        # Use the correct scaling factor
        fprj_batch = fprj_batch * fprj_batch.shape[1] / projector_batch.rotview

        img_batch = fbp_bp(
            projector_batch,
            fprj_batch,
            theta0_batch,
            zrot,
            m_pi,
            q_value
        )

        img.append(img_batch)
    img = np.concatenate(img, 1)

    return img
