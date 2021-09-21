import numpy as np
import cupy as cp
from ctypes import cdll, c_int, c_void_p, c_float, c_ulong
import os
from scipy.ndimage.filters import gaussian_filter

module = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libprior.so'))


def nlm(img, guide, d, search_size, kernel_size, kernel_std, eps=1e-6):
    '''
    Non local mean denoising with guide.

    @params:
    @img - the image to be denoised
    @guide - the guide image for nlm.
    @d - larger the d, the weaker the guide. d should be estimated based on the noise level of guide. 
    @search_size - the search window size for averaging
    @kernel_size - the gaussian kernel size to calculate distance between two points.
    @kernel_std - std of the gaussian kernel
    @eps - regularization

    @return:
    res - the denoised image
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
