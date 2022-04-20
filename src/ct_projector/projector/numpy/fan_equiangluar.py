'''
Equiangular fanbeam geometry
'''

from ctypes import cdll, POINTER, c_float, c_int, c_ulong
from typing import Union
import numpy as np

from .ct_projector import ct_projector

import pkg_resources

module = cdll.LoadLibrary(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libprojector.so')
)


# %%
def ramp_filter(projector: ct_projector, prj: np.array, filter_type: str = 'hann') -> np.array:
    '''
    Filter the projection for equiangular geometry.

    Parameters
    ---------------
    projector: ct_projector.
        The parameter wrapper.
    prj: np.array(float32) of shape [batch, nview, nv, nu].
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

    prj = prj.astype(np.float32)
    fprj = np.zeros(prj.shape, np.float32)

    module.cfbpFanFilter.restype = c_int

    err = module.cfbpFanFilter(
        fprj.ctypes.data_as(POINTER(c_float)),
        prj.ctypes.data_as(POINTER(c_float)),
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


def fbp_bp(projector: ct_projector, prj: np.array, angles: np.array) -> np.array:
    '''
    Fanbeam backprojection with circular equiangular detector. Ray driven
    and weighted for FBP.

    Parameters
    ----------------
    prj: np.array(float32) of size [batch, nview, nv, nu].
        The projection to be backprojected. The size does not need to be the same
        with projector.nview, projector.nv, projector.nu.
    angles: np.array(float32) of size [nview].
        The projection angles in radius.

    Returns
    --------------
    img: np.array(float32) of size [batch, projector.nz, projector.ny, projector.nx]
        The backprojected image.
    '''

    prj = prj.astype(np.float32)
    angles = angles.astype(np.float32)
    img = np.zeros([prj.shape[0], projector.nz, projector.ny, projector.nx], np.float32)

    module.cfbpFanBackprojection.restype = c_int

    err = module.cfbpFanBackprojection(
        img.ctypes.data_as(POINTER(c_float)),
        prj.ctypes.data_as(POINTER(c_float)),
        angles.ctypes.data_as(POINTER(c_float)),
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


def siddon_fp(projector: ct_projector, img: np.array, angles: np.array) -> np.array:
    '''
    Fanbeam forward projection with circular equiangular detector. Siddon ray driven.

    Parameters
    ----------------
    img: np.array(float32) of size [batch, nz, ny, nx]
        The image to be projected.
    angles: np.array(float32) of size [nview]
        The projection angles in radius.

    Returns
    --------------
    prj: np.array(float32) of size [batch, projector.nview, projector.nv, projector.nu]
        The forward projection.
    '''

    img = img.astype(np.float32)
    angles = angles.astype(np.float32)
    prj = np.zeros([img.shape[0], len(angles), projector.nv, projector.nu], np.float32)

    module.cSiddonFanProjection.restype = c_int

    err = module.cSiddonFanProjection(
        prj.ctypes.data_as(POINTER(c_float)),
        img.ctypes.data_as(POINTER(c_float)),
        angles.ctypes.data_as(POINTER(c_float)),
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

    return prj


def siddon_bp(projector: ct_projector, prj: np.array, angles: np.array) -> np.array:
    '''
    Fanbeam backprojection with circular equiangular detector. Siddon ray driven.

    Parameters
    ----------------
    prj: np.array(float32) of size [batch, nview, nv, nu].
        The projection to be backprojected. The size does not need to be the same
        with projector.nview, projector.nv, projector.nu.
    angles: np.array(float32) of size [nview].
        The projection angles in radius.

    Returns
    --------------
    img: np.array(float32) of size [batch, projector.nz, projector.ny, projector.nx]
        The backprojected image.
    '''

    prj = prj.astype(np.float32)
    angles = angles.astype(np.float32)
    img = np.zeros([prj.shape[0], projector.nz, projector.ny, projector.nx], np.float32)

    module.cSiddonFanBackprojection.restype = c_int

    err = module.cSiddonFanBackprojection(
        img.ctypes.data_as(POINTER(c_float)),
        prj.ctypes.data_as(POINTER(c_float)),
        angles.ctypes.data_as(POINTER(c_float)),
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
    img: np.array,
    angles: np.array,
    branchless: Union[bool, int] = False
) -> np.array:
    '''
    Fanbeam forward projection with circular equiangular detector. Distance driven.

    Parameters
    ----------------
    img: array(float32) of size [batch, nz, ny, nx]
        The image to be projected.
    angles: array(float32) of size [nview]
        The projection angles in radius.

    Returns
    --------------
    prj: array(float32) of size [batch, projector.nview, projector.nv, projector.nu]
        The forward projection.
    '''
    type_projector = 0
    if branchless:
        type_projector += 2

    prj = np.zeros([img.shape[0], len(angles), projector.nv, projector.nu], np.float32)

    module.cDistanceDrivenFanProjection.restype = c_int

    err = module.cDistanceDrivenFanProjection(
        prj.ctypes.data_as(POINTER(c_float)),
        img.ctypes.data_as(POINTER(c_float)),
        angles.ctypes.data_as(POINTER(c_float)),
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
    prj: np.array,
    angles: np.array,
    is_fbp: Union[bool, int] = False,
    branchless: Union[bool, int] = False
) -> np.array:
    '''
    Fanbeam backprojection with circular equiangular detector. Distance driven.

    Parameters
    ----------------
    prj: array(float32) of size [batch, nview, nv, nu].
        The projection to be backprojected. The size does not need to be the same
        with projector.nview, projector.nv, projector.nu.
    angles: array(float32) of size [nview].
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

    img = np.zeros([prj.shape[0], projector.nz, projector.ny, projector.nx], np.float32)

    module.cDistanceDrivenFanBackprojection.restype = c_int

    err = module.cDistanceDrivenFanBackprojection(
        img.ctypes.data_as(POINTER(c_float)),
        prj.ctypes.data_as(POINTER(c_float)),
        angles.ctypes.data_as(POINTER(c_float)),
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
