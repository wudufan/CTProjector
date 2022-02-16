from ctypes import cdll, c_int, c_void_p, c_ulong, c_float

import cupy as cp

from .ct_projector import ct_projector

import pkg_resources

module = cdll.LoadLibrary(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libprojector.so')
)


# %%
def siddon_fp_arbitrary(
    projector: ct_projector,
    img: cp.array,
    det_center: cp.array,
    det_u: cp.array,
    det_v: cp.array,
    src: cp.array
) -> cp.array:
    '''
    Conebeam forward projection with arbitrary geometry and flat panel. Using Siddon ray tracing.
    The size of the img will override that of projector.nx, projector.ny, projector.nz. The projection size
    will be [batch, nview, projector.nv, projector.nu].

    Parameters
    -------------------------
    projector: ct_projector class.
        The ct_projector class with all the parameters.
    img: cp.array(float32) of size [batch, nz, ny, nx].
        The image to be projected, nz, ny, nx can be different than projector.nz, projector.ny, projector.nx.
        The projector will always use the size of the image.
    det_center: cp.array(float32) of size [nview, 3].
        The center of the detector in mm. Each row records the center of detector as (z, y, x).
    det_u: cp.array(float32) of size [nview, 3].
        The u axis of the detector. Each row is a normalized vector in (z, y, x).
    det_v: cp.array(float32) of size [nview ,3].
        The v axis of the detector. Each row is a normalized vector in (z, y, x).
    src: cp.array(float32) of size [nview, 3].
        The src positions in mm. Each row records in the source position as (z, y, x).

    Returns
    -------------------------
    prj: cp.array(float32) of size [batch, projector.nview, projector.nv, projector.nu].
        The forward projection.
    '''

    # projection of size
    prj = cp.zeros([img.shape[0], det_center.shape[0], projector.nv, projector.nu], cp.float32)

    module.cupySiddonConeProjectionArbitrary.restype = c_int

    err = module.cupySiddonConeProjectionArbitrary(
        c_void_p(prj.data.ptr),
        c_void_p(img.data.ptr),
        c_void_p(det_center.data.ptr),
        c_void_p(det_u.data.ptr),
        c_void_p(det_v.data.ptr),
        c_void_p(src.data.ptr),
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
    prj: cp.array,
    det_center: cp.array,
    det_u: cp.array,
    det_v: cp.array,
    src: cp.array
) -> cp.array:
    '''
    Conebeam backprojection with arbitrary geometry and flat panel. Using Siddon ray tracing.
    The size of the img will override that of projector.nx, projector.ny, projector.nz. The projection size
    will be [batch, nview, projector.nv, projector.nu].

    Parameters
    -------------------------
    projector: ct_projector class.
        The ct_projector class with all the parameters.
    prj: cp.array(float32) of size [batch, nview, nv, nu].
        The projection to be backprojected. It will override the default shape predefined,
        i.e. projector.nview, projector.nv, projector.nu.
    det_center: cp.array(float32) of size [nview, 3].
        The center of the detector in mm. Each row records the center of detector as (z, y, x).
    det_u: cp.array(float32) of size [nview, 3].
        The u axis of the detector. Each row is a normalized vector in (z, y, x).
    det_v: cp.array(float32) of size [nview ,3].
        The v axis of the detector. Each row is a normalized vector in (z, y, x).
    src: cp.array(float32) of size [nview, 3].
        The src positions in mm. Each row records in the source position as (z, y, x).

    Returns
    -------------------------
    img: cp.array(float32) of size [batch, projector.nz, projector.ny, projector.nx].
        The backprojected image.
    '''

    # make sure they are float32
    # projection of size
    img = cp.zeros([prj.shape[0], projector.nz, projector.ny, projector.nx], cp.float32)

    module.cupySiddonConeBackprojectionArbitrary.restype = c_int

    err = module.cupySiddonConeBackprojectionArbitrary(
        c_void_p(img.data.ptr),
        c_void_p(prj.data.ptr),
        c_void_p(det_center.data.ptr),
        c_void_p(det_u.data.ptr),
        c_void_p(det_v.data.ptr),
        c_void_p(src.data.ptr),
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
