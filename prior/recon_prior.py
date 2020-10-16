import numpy as np
from ctypes import *
import os
from scipy.ndimage.filters import gaussian_filter

module = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libprior.so'))

def set_device(device):
    module.cSetDevice.restype = c_int
    return module.cSetDevice(c_int(device))

def nlm(img, guide, d, search_size, kernel_size, kernel_std, eps = 1e-6):
    '''
    Non local mean denoising with guide.

    @params:
    @img - the image to be denoised
    @guide - the guide image for nlm
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
    
    res = np.zeros(img.shape, np.float32)
    img = img.astype(np.float32)
    guide = guide.astype(np.float32)
    kernel = kernel.astype(np.float32)
    
    module.cNlm.restype = c_int
    err = module.cNlm(res.ctypes.data_as(POINTER(c_float)), 
                      img.ctypes.data_as(POINTER(c_float)), 
                      guide.ctypes.data_as(POINTER(c_float)), 
                      kernel.ctypes.data_as(POINTER(c_float)), 
                      c_float(d*d), c_float(eps),
                      c_int(search_size[2]), c_int(search_size[1]), c_int(search_size[0]),
                      c_ulong(img.shape[0]), c_ulong(img.shape[3]), c_ulong(img.shape[2]), c_ulong(img.shape[1]), 
                      c_int(kernel_size[2]), c_int(kernel_size[1]), c_int(kernel_size[0]))
    
    if not err == 0:
        print (err)
    
    return res