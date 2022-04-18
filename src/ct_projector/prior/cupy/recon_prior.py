'''
Cupy wrapper of the denoisers.
'''

import numpy as np
import cupy as cp
from ctypes import cdll, c_int, c_void_p, c_float, c_ulong
from typing import Tuple
from scipy.ndimage.filters import gaussian_filter

import pkg_resources

module = cdll.LoadLibrary(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libprior.so')
)


def nlm(
    img: cp.array,
    guide: cp.array,
    d: float,
    search_size: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int],
    kernel_std: float,
    eps: float = 1e-6
) -> cp.array:
    '''
    Non local mean denoising with guide.

    Parameters
    -----------------
    img: cp.array(float32) of shape [batch, nz, ny, nx].
        The image to be denoised.
    guide: cp.array(float32) of shape [batch, nz, ny, nx].
        The guide image for NLM.
    d: float.
        The larger the d, the weaker the guide. d should be estimated based on the noise level of guide.
    search_size: tuple of length 3.
        The search window size for averaging.
    kernel_size: tuple of length 3.
        the gaussian kernel size to calculate distance between two points.
    kernel_std: float.
        Std of the gaussian kernel
    eps: float.
        Regularization factor.

    Returns
    -------------------
    res: cp.array(float) of shape [batch, nz, ny, nx].
        The denoised image.
    '''
    kernel = np.zeros(kernel_size, np.float32)
    kernel[int(kernel_size[0] / 2), int(kernel_size[1] / 2), int(kernel_size[2] / 2)] = 1
    kernel = gaussian_filter(kernel, kernel_std)
    kernel = cp.array(kernel, dtype=cp.float32, order='C')

    res = cp.zeros(img.shape, cp.float32)

    module.cupyNlm.restype = c_int
    err = module.cupyNlm(
        c_void_p(res.data.ptr),
        c_void_p(img.data.ptr),
        c_void_p(guide.data.ptr),
        c_void_p(kernel.data.ptr),
        c_float(d * d),
        c_float(eps),
        c_int(search_size[2]),
        c_int(search_size[1]),
        c_int(search_size[0]),
        c_ulong(img.shape[0]),
        c_ulong(img.shape[3]),
        c_ulong(img.shape[2]),
        c_ulong(img.shape[1]),
        c_int(kernel_size[2]),
        c_int(kernel_size[1]),
        c_int(kernel_size[0])
    )

    if not err == 0:
        print(err)

    return res


def tv_sqs(
    img: cp.array,
    weights: Tuple[float, float, float],
    eps: float = 1e-8
):
    '''
    The TV prior for SQS algorithm.

    Parameters
    -----------------
    img: cp.array(float32) of shape [batch, nz, ny, nx].
        The image to be calculated for TV prior.
    weights: tuple of length 3.
        The weighting factor along z, y, x.
    eps: float.
        The normalization at x = 0.

    Returns
    ------------------
    s1: cp.array(float32) of shape [batch, nz, ny, nx].
        The first-order derivative of the prior.
    s2: cp.array(float32) of shape [batch, nz, ny, nx].
        The second-order derivative of the prior.
    var: cp.array(float32) of shape [batch, nz, ny, nx].
        The variation map.
    '''

    module.cupyTVSQS3D.restype = c_int
    s1 = cp.zeros(img.shape, cp.float32)
    s2 = cp.zeros(img.shape, cp.float32)
    var = cp.zeros(img.shape, cp.float32)

    err = module.cupyTVSQS3D(
        c_void_p(s1.data.ptr),
        c_void_p(s2.data.ptr),
        c_void_p(var.data.ptr),
        c_void_p(img.data.ptr),
        c_float(weights[2]),
        c_float(weights[1]),
        c_float(weights[0]),
        c_ulong(img.shape[0]),
        c_ulong(img.shape[3]),
        c_ulong(img.shape[2]),
        c_ulong(img.shape[1]),
        c_float(eps)
    )

    if not err == 0:
        print(err)

    return s1, s2, var
