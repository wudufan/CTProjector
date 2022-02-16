'''
Conebeam geometry
'''

from ctypes import cdll, POINTER, c_float, c_int, c_ulong

import numpy as np

from .ct_projector import ct_projector

import pkg_resources

module = cdll.LoadLibrary(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libprojector.so')
)


# %%
def siddon_fp_arbitrary(
    projector: ct_projector,
    img: np.array,
    det_center: np.array,
    det_u: np.array,
    det_v: np.array,
    src: np.array
) -> np.array:
    '''
    Conebeam forward projection with arbitrary geometry and flat panel. Using Siddon ray tracing.
    The size of the img will override that of projector.nx, projector.ny, projector.nz. The projection size
    will be [batch, nview, projector.nv, projector.nu].

    Parameters
    -------------------------
    img: np.array(float32) of size [batch, nz, ny, nx].
        The image to be projected, nz, ny, nx can be different than projector.nz, projector.ny, projector.nx.
        The projector will always use the size of the image.
    det_center: np.array(float32) of size [nview, 3].
        The center of the detector in mm. Each row records the center of detector as (z, y, x).
    det_u: np.array(float32) of size [nview, 3].
        The u axis of the detector. Each row is a normalized vector in (z, y, x).
    det_v: np.array(float32) of size [nview ,3].
        The v axis of the detector. Each row is a normalized vector in (z, y, x).
    src: np.array(float32) of size [nview, 3].
        The src positions in mm. Each row records in the source position as (z, y, x).

    Returns
    -------------------------
    prj: np.array(float32) of size [batch, projector.nview, projector.nv, projector.nu].
        The forward projection.
    '''

    # make sure they are float32
    img = img.astype(np.float32)
    det_center = det_center.astype(np.float32)
    det_u = det_u.astype(np.float32)
    det_v = det_v.astype(np.float32)
    src = src.astype(np.float32)

    # projection of size
    prj = np.zeros([img.shape[0], det_center.shape[0], projector.nv, projector.nu], np.float32)

    module.cSiddonConeProjectionArbitrary.restype = c_int

    err = module.cSiddonConeProjectionArbitrary(
        prj.ctypes.data_as(POINTER(c_float)),
        img.ctypes.data_as(POINTER(c_float)),
        det_center.ctypes.data_as(POINTER(c_float)),
        det_u.ctypes.data_as(POINTER(c_float)),
        det_v.ctypes.data_as(POINTER(c_float)),
        src.ctypes.data_as(POINTER(c_float)),
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
        c_float(projector.du),
        c_float(projector.dv),
        c_float(projector.off_u),
        c_float(projector.off_v)
    )

    if err != 0:
        print(err)

    return prj


def siddon_bp_arbitrary(
    projector: ct_projector,
    prj: np.array,
    det_center: np.array,
    det_u: np.array,
    det_v: np.array,
    src: np.array
) -> np.array:
    '''
    Conebeam backprojection with arbitrary geometry and flat panel. Using Siddon ray tracing.
    The size of the img will override that of projector.nx, projector.ny, projector.nz. The projection size
    will be [batch, nview, projector.nv, projector.nu].

    Parameters
    -------------------------
    prj: np.array(float32) of size [batch, nview, nv, nu].
        The projection to be backprojected. It will override the default shape predefined,
        i.e. projector.nview, projector.nv, projector.nu.
    det_center: np.array(float32) of size [nview, 3].
        The center of the detector in mm. Each row records the center of detector as (z, y, x).
    det_u: np.array(float32) of size [nview, 3].
        The u axis of the detector. Each row is a normalized vector in (z, y, x).
    det_v: np.array(float32) of size [nview ,3].
        The v axis of the detector. Each row is a normalized vector in (z, y, x).
    src: np.array(float32) of size [nview, 3].
        The src positions in mm. Each row records in the source position as (z, y, x).

    Returns
    -------------------------
    img: np.array(float32) of size [batch, projector.nz, projector.ny, projector.nx].
        The backprojected image.
    '''

    # make sure they are float32
    prj = prj.astype(np.float32)
    det_center = det_center.astype(np.float32)
    det_u = det_u.astype(np.float32)
    det_v = det_v.astype(np.float32)
    src = src.astype(np.float32)

    # projection of size
    img = np.zeros([prj.shape[0], projector.nz, projector.ny, projector.nx], np.float32)

    module.cSiddonConeBackprojectionArbitrary.restype = c_int

    err = module.cSiddonConeBackprojectionArbitrary(
        img.ctypes.data_as(POINTER(c_float)),
        prj.ctypes.data_as(POINTER(c_float)),
        det_center.ctypes.data_as(POINTER(c_float)),
        det_u.ctypes.data_as(POINTER(c_float)),
        det_v.ctypes.data_as(POINTER(c_float)),
        src.ctypes.data_as(POINTER(c_float)),
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
        c_float(projector.du),
        c_float(projector.dv),
        c_float(projector.off_u),
        c_float(projector.off_v)
    )

    if err != 0:
        print(err)

    return img
