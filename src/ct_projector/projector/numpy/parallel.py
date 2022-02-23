'''
Parallel beam geometry
'''

# %%
from ctypes import cdll, POINTER, c_float, c_int, c_ulong

import numpy as np

from .ct_projector import ct_projector

import pkg_resources

module = cdll.LoadLibrary(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libprojector.so')
)


# %%
def ramp_filter(projector: ct_projector, prj: np.array, filter_type: str = 'hann') -> np.array:
    '''
    Filter the projection for parallel geometry.

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

    module.cfbpParallelFilter.restype = c_int

    err = module.cfbpParallelFilter(
        fprj.ctypes.data_as(POINTER(c_float)),
        prj.ctypes.data_as(POINTER(c_float)),
        c_ulong(prj.shape[0]),
        c_ulong(prj.shape[3]),
        c_ulong(prj.shape[2]),
        c_ulong(prj.shape[1]),
        c_float(projector.du),
        c_float(projector.dv),
        c_float(projector.off_u),
        c_float(projector.off_v),
        c_int(ifilter)
    )

    if err != 0:
        print(err)

    return fprj
