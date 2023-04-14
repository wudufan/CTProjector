'''
Numpy Wrapper of the prior functions.
'''

from typing import Tuple
import numpy as np
from ctypes import cdll, c_int, POINTER, c_float, c_ulong
from scipy.ndimage.filters import gaussian_filter

import pkg_resources

module = cdll.LoadLibrary(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libprior.so')
)


def set_device(device: int) -> int:
    '''
    Set which GPU to use.
    '''
    module.cSetDevice.restype = c_int
    return module.cSetDevice(c_int(device))


def nlm(
    img: np.array,
    guide: np.array,
    d: float,
    search_size: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int],
    kernel_std: float,
    eps: float = 1e-6
) -> np.array:
    '''
    Non local mean denoising with guide.

    Parameters
    -----------------
    img: np.array(float32) of shape [batch, nz, ny, nx].
        The image to be denoised.
    guide: np.array(float32) of shape [batch, nz, ny, nx].
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
    res: np.array(float) of shape [batch, nz, ny, nx].
        The denoised image.
    '''
    kernel = np.zeros(kernel_size, np.float32)
    kernel[int(kernel_size[0] / 2), int(kernel_size[1] / 2), int(kernel_size[2] / 2)] = 1
    kernel = gaussian_filter(kernel, kernel_std)

    res = np.zeros(img.shape, np.float32)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if guide.dtype != np.float32:
        guide = guide.astype(np.float32)
    if kernel.dtype != np.float32:
        kernel = kernel.astype(np.float32)

    module.cNlm.restype = c_int
    err = module.cNlm(
        res.ctypes.data_as(POINTER(c_float)),
        img.ctypes.data_as(POINTER(c_float)),
        guide.ctypes.data_as(POINTER(c_float)),
        kernel.ctypes.data_as(POINTER(c_float)),
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


def hypr_nlm(
    img: np.array,
    guide: np.array,
    d: float,
    search_size: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int],
    kernel_std: float,
    eps: float = 1e-6
) -> np.array:
    '''
    Non local mean denoising with guide.

    Parameters
    -----------------
    img: np.array(float32) of shape [batch, nz, ny, nx].
        The image to be denoised.
    guide: np.array(float32) of shape [batch, nz, ny, nx].
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
    res: np.array(float) of shape [batch, nz, ny, nx].
        The denoised image.
    '''
    kernel = np.zeros(kernel_size, np.float32)
    kernel[int(kernel_size[0] / 2), int(kernel_size[1] / 2), int(kernel_size[2] / 2)] = 1
    kernel = gaussian_filter(kernel, kernel_std)

    res = np.zeros(img.shape, np.float32)
    img = img.astype(np.float32)
    guide = guide.astype(np.float32)
    kernel = kernel.astype(np.float32)

    module.cHyprNlm.restype = c_int
    err = module.cHyprNlm(
        res.ctypes.data_as(POINTER(c_float)),
        img.ctypes.data_as(POINTER(c_float)),
        guide.ctypes.data_as(POINTER(c_float)),
        kernel.ctypes.data_as(POINTER(c_float)),
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
    img: np.array,
    weights: Tuple[float, float, float],
    eps: float = 1e-8
):
    '''
    The TV prior for SQS algorithm.

    Parameters
    -----------------
    img: np.array(float32) of shape [batch, nz, ny, nx].
        The image to be calculated for TV prior.
    weights: tuple of length 3.
        The weighting factor along z, y, x.
    eps: float.
        The normalization at x = 0.

    Returns
    ------------------
    s1: np.array(float32) of shape [batch, nz, ny, nx].
        The first-order derivative of the prior.
    s2: np.array(float32) of shape [batch, nz, ny, nx].
        The second-order derivative of the prior.
    var: np.array(float32) of shape [batch, nz, ny, nx].
        The variation map.
    '''

    module.cTVSQS3D.restype = c_int
    s1 = np.zeros(img.shape, np.float32)
    s2 = np.zeros(img.shape, np.float32)
    var = np.zeros(img.shape, np.float32)

    err = module.cTVSQS3D(
        s1.ctypes.data_as(POINTER(c_float)),
        s2.ctypes.data_as(POINTER(c_float)),
        var.ctypes.data_as(POINTER(c_float)),
        img.ctypes.data_as(POINTER(c_float)),
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
