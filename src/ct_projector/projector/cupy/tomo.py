from ctypes import cdll, c_int, c_void_p, c_ulong, c_float

import cupy as cp

from .ct_projector import ct_projector

import pkg_resources

module = cdll.LoadLibrary(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libprojector.so')
)


# %%
def distance_driven_fp(
    projector: ct_projector,
    img: cp.array,
    det_center: cp.array,
    src: cp.array,
    branchless: bool = False
) -> cp.array:
    '''
    Distance driven forward projection for tomosynthesis. It assumes that the detector has
    u=(1,0,0) and v = (0,1,0).
    The projection should be along the z-axis (main axis for distance driven projection).
    The img size will override the default predefined shape. The forward projection shape
    will be [batch, projector.nview, projector.nv, projector.nu].

    Parameters
    -------------------
    img: cp.array(float32) of size (batch, nz, ny, nx).
        The image to be projected, nz, ny, nx can be different than projector.nz, projector.ny, projector.nx.
        The projector will always use the size of the image.
    det_center: cp.array(float32) of size [nview, 3].
        The center of the detector in mm. Each row records the center of detector as (z, y, x).
    src: cp.array(float32) of size [nview, 3].
        The src positions in mm. Each row records in the source position as (z, y, x).
    branchless: bool
        If True, use the branchless mode (double precision required).

    Returns
    -------------------------
    prj: cp.array(float32) of size [batch, projector.nview, projector.nv, projector.nu].
        The forward projection.
    '''
    prj = cp.zeros([img.shape[0], det_center.shape[0], projector.nv, projector.nu], cp.float32)

    if branchless:
        type_projector = 1
    else:
        type_projector = 0

    module.cupyDistanceDrivenTomoProjection.restype = c_int
    err = module.cupyDistanceDrivenTomoProjection(
        c_void_p(prj.data.ptr),
        c_void_p(img.data.ptr),
        c_void_p(det_center.data.ptr),
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
        c_float(projector.off_v),
        c_int(type_projector)
    )

    if err != 0:
        print(err)

    return prj


def distance_driven_bp(
    projector: ct_projector,
    prj: cp.array,
    det_center: cp.array,
    src: cp.array,
    branchless: bool = False
) -> cp.array:
    '''
    Distance driven backprojection for tomosynthesis. It assumes that the detector has
    u=(1,0,0) and v = (0,1,0).
    The backprojection should be along the z-axis (main axis for distance driven projection).
    The size of the img will override that of projector.nx, projector.ny, projector.nz. The projection size
    will be [batch, nview, projector.nv, projector.nu].

    Parameters
    -------------------------
    prj: cp.array(float32) of size [batch, nview, nv, nu].
        The projection to be backprojected. It will override the default shape predefined,
        i.e. projector.nview, projector.nv, projector.nu.
    det_center: cp.array(float32) of size [nview, 3].
        The center of the detector in mm. Each row records the center of detector as (z, y, x).
    src: cp.array(float32) of size [nview, 3].
        The src positions in mm. Each row records in the source position as (z, y, x).
    branchless: bool
        If True, use the branchless mode (double precision required).

    Returns
    -------------------------
    img: cp.array(float32) of size [batch, projector.nz, projector.ny, projector.nx].
        The backprojected image.
    '''

    img = cp.zeros([prj.shape[0], projector.nz, projector.ny, projector.nx], cp.float32)

    if branchless:
        type_projector = 1
    else:
        type_projector = 0

    module.cupyDistanceDrivenTomoBackprojection.restype = c_int
    err = module.cupyDistanceDrivenTomoBackprojection(
        c_void_p(img.data.ptr),
        c_void_p(prj.data.ptr),
        c_void_p(det_center.data.ptr),
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
        c_float(projector.off_v),
        c_int(type_projector)
    )

    if err != 0:
        print(err)

    return img


# %%
def filter_fbp(
    projector: ct_projector,
    prj: cp.array,
    det_center: cp.array,
    src: cp.array,
    filter_type: str = 'hann',
    cutoff_x: float = 1,
    cutoff_y: float = 1
) -> cp.array:
    '''
    Filter of the projection for tomosynthesis reconstruction.

    Parameters
    ------------------
    prj: cp.array(float32) of size [batch, nview, nv, nu].
        The projection to be filtered. It will override the default shape predefined,
        i.e. projector.nview, projector.nv, projector.nu.
    det_center: cp.array(float32) of size [nview, 3].
        The center of the detector in mm. Each row records the center of detector as (z, y, x).
    src: cp.array(float32) of size [nview, 3].
        The src positions in mm. Each row records in the source position as (z, y, x).
    filter_type: str (case insensitive).
        'hamming', 'hann', or other. If other than 'hamming' or 'hann', use ramp filter.
    cutoff_x: float.
        Cutoff frequency along u direction, value between (0, 1].
    cutoff_y: float.
        Cutoff frequency along v direction, value between (0, 1].

    Returns
    ------------------
    fprj: cp.array(float32) of shape [batch, nview, nv, nu].
        The filtered projection.
    '''
    if filter_type.lower() == 'hamming':
        ifilter = 1
    elif filter_type.lower() == 'hann':
        ifilter = 2
    else:
        ifilter = 0

    fprj = cp.zeros(prj.shape, cp.float32)

    module.cupyFbpTomoFilter.restype = c_int
    err = module.cupyFbpTomoFilter(
        c_void_p(fprj.data.ptr),
        c_void_p(prj.data.ptr),
        c_void_p(det_center.data.ptr),
        c_void_p(src.data.ptr),
        c_ulong(prj.shape[0]),
        c_ulong(prj.shape[3]),
        c_ulong(prj.shape[2]),
        c_ulong(prj.shape[1]),
        c_float(projector.du),
        c_float(projector.dx),
        c_float(projector.dz),
        c_int(ifilter),
        c_float(cutoff_x),
        c_float(cutoff_y)
    )

    if err != 0:
        print(err)

    return fprj
