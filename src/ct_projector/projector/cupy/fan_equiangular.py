'''
Equiangular fanbeam geometry
'''

from ctypes import cdll, c_int, c_void_p, c_ulong, c_float
from typing import Union
import cupy as cp

from .ct_projector import ct_projector

import pkg_resources

module = cdll.LoadLibrary(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libprojector.so')
)


# %%
def ramp_filter(projector: ct_projector, prj: cp.array, filter_type: str = 'hann') -> cp.array:
    '''
    Filter the projection for equiangular geometry.

    Parameters
    ---------------
    projector: ct_projector.
        The parameter wrapper.
    prj: array(float32) of shape [batch, nview, nv, nu].
        The projection to be filtered. The shape will override the default ones,
        projector.nview, projector.nv, projector.nu.
    filter_type: str.
        Possible values are 'hamming', 'hann', 'cosine'. If not the above values,
        will use RL filter.

    Returns
    --------------
    fprj: np.array(float32) of shape [batch, nview, nv, nu].
        The filtered projection.
    '''
    if filter_type.lower() == 'hamming':
        ifilter = 1
    elif filter_type.lower() == 'hann':
        ifilter = 2
    elif filter_type.lower() == 'cosine':
        ifilter = 3
    else:
        ifilter = 0

    fprj = cp.zeros(prj.shape, cp.float32)

    module.cupyfbpFanFilter.restype = c_int

    err = module.cupyfbpFanFilter(
        c_void_p(fprj.data.ptr),
        c_void_p(prj.data.ptr),
        c_ulong(prj.shape[0]),
        c_ulong(prj.shape[3]),
        c_ulong(prj.shape[2]),
        c_ulong(prj.shape[1]),
        c_float(projector.du / projector.dsd),
        c_float(projector.dv),
        c_float(projector.off_u),
        c_float(projector.off_v),
        c_float(projector.dsd),
        c_float(projector.dso),
        c_int(ifilter)
    )

    if err != 0:
        print(err)

    return fprj


def fbp_bp(projector: ct_projector, prj: cp.array, angles: cp.array) -> cp.array:
    '''
    Fanbeam backprojection with circular equiangular detector. Ray driven
    and weighted for FBP.

    Parameters
    ----------------
    prj: array(float32) of size [batch, nview, nv, nu].
        The projection to be backprojected. The size does not need to be the same
        with projector.nview, projector.nv, projector.nu.
    angles: array(float32) of size [nview].
        The projection angles in radius.

    Returns
    --------------
    img: np.array(float32) of size [batch, projector.nz, projector.ny, projector.nx]
        The backprojected image.
    '''

    img = cp.zeros([prj.shape[0], projector.nz, projector.ny, projector.nx], cp.float32)

    module.cupyfbpFanBackprojection.restype = c_int

    err = module.cfbpFanBackprojection(
        c_void_p(img.data.ptr),
        c_void_p(prj.data.ptr),
        c_void_p(angles.data.ptr),
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
        c_ulong(prj.shape[3]),
        c_ulong(prj.shape[2]),
        c_ulong(prj.shape[1]),
        c_float(projector.du / projector.dsd),
        c_float(projector.dv),
        c_float(projector.off_u),
        c_float(projector.off_v),
        c_float(projector.dsd),
        c_float(projector.dso)
    )

    if err != 0:
        print(err)

    return img


# %%
def distance_driven_fp(
    projector: ct_projector,
    img: cp.array,
    angles: cp.array,
    branchless: Union[bool, int] = False
) -> cp.array:
    '''
    Fanbeam forward projection with circular equiangular detector. Distance driven.

    Parameters
    ----------------
    img: cp.array(float32) of size [batch, nz, ny, nx]
        The image to be projected.
    angles: cp.array(float32) of size [nview]
        The projection angles in radius.
    branchless: boolean
        If true, use branchless distance driven projector

    Returns
    --------------
    prj: cp.array(float32) of size [batch, projector.nview, projector.nv, projector.nu]
        The forward projection.
    '''
    type_projector = 0
    if branchless:
        type_projector += 2

    prj = cp.zeros([img.shape[0], len(angles), projector.nv, projector.nu], cp.float32)

    module.cupyDistanceDrivenFanProjection.restype = c_int

    err = module.cupyDistanceDrivenFanProjection(
        c_void_p(prj.data.ptr),
        c_void_p(img.data.ptr),
        c_void_p(angles.data.ptr),
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
        c_ulong(prj.shape[3]),
        c_ulong(prj.shape[2]),
        c_ulong(prj.shape[1]),
        c_float(projector.du / projector.dsd),
        c_float(projector.dv),
        c_float(projector.off_u),
        c_float(projector.off_v),
        c_float(projector.dsd),
        c_float(projector.dso),
        c_int(type_projector)
    )

    if err != 0:
        print(err)

    return prj


def distance_driven_bp(
    projector: ct_projector,
    prj: cp.array,
    angles: cp.array,
    is_fbp: Union[bool, int] = False,
    branchless: Union[bool, int] = False
) -> cp.array:
    '''
    Fanbeam backprojection with circular equiangular detector. Distance driven.

    Parameters
    ----------------
    prj: cp.array(float32) of size [batch, nview, nv, nu].
        The projection to be backprojected. The size does not need to be the same
        with projector.nview, projector.nv, projector.nu.
    angles: cp.array(float32) of size [nview].
        The projection angles in radius.
    is_fbp: bool.
        if true, use the FBP weighting scheme to backproject filtered data.

    Returns
    --------------
    img: cp.array(float32) of size [batch, projector.nz, projector.ny, projector.nx]
        The backprojected image.
    '''
    type_projector = 0
    if is_fbp:
        type_projector += 1
    if branchless:
        type_projector += 2

    img = cp.zeros([prj.shape[0], projector.nz, projector.ny, projector.nx], cp.float32)

    module.cupyDistanceDrivenFanBackprojection.restype = c_int

    err = module.cupyDistanceDrivenFanBackprojection(
        c_void_p(img.data.ptr),
        c_void_p(prj.data.ptr),
        c_void_p(angles.data.ptr),
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
        c_ulong(prj.shape[3]),
        c_ulong(prj.shape[2]),
        c_ulong(prj.shape[1]),
        c_float(projector.du / projector.dsd),
        c_float(projector.dv),
        c_float(projector.off_u),
        c_float(projector.off_v),
        c_float(projector.dsd),
        c_float(projector.dso),
        c_int(type_projector)
    )

    if err != 0:
        print(err)

    return img
