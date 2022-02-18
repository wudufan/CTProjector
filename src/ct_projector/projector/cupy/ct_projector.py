'''
Cupy-based wrapper of the CUDA functions. Compared to numpy wrapper, it reduces memory transfer
between CPU and GPU but costs more GPU memory.
'''

from ctypes import cdll, c_int
from typing import Callable

import cupy as cp
import numpy as np
import configparser

import pkg_resources

module = cdll.LoadLibrary(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libprojector.so')
)


def set_device(device: int) -> int:
    '''
    Set the computing device and return any error code.
    '''
    return module.SetDevice(c_int(device))


class ct_projector:
    '''
    CT projector wrapper
    '''
    def __init__(self):
        self.nview = 720
        self.rotview = 720
        self.nu = 512
        self.nv = 512
        self.nx = 512
        self.ny = 512
        self.nz = 512
        self.dx = 1
        self.dy = 1
        self.dz = 1
        self.cx = 0
        self.cy = 0
        self.cz = 0
        self.dsd = 1085.6
        self.dso = 595
        self.du = 1.2
        self.dv = 1.2
        self.off_u = 0
        self.off_v = 0

    def from_file(self, filename: str) -> None:
        '''
        Load the geometry from config filename.
        '''
        self.geometry = configparser.ConfigParser()
        _ = self.geometry.read(filename)

        for o in self.geometry['geometry']:
            if '_mm' in o:
                setattr(self, o[:-3], np.float32(self.geometry['geometry'][o]))
            else:
                setattr(self, o, np.float32(self.geometry['geometry'][o]))

        self.nview = int(self.nview)
        self.nu = int(self.nu)
        self.nv = int(self.nv)
        self.nx = int(self.nx)
        self.ny = int(self.ny)
        self.nz = int(self.nz)

    def set_projector(self, projector: Callable[..., cp.array], **kwargs) -> None:
        '''
        Set the projector. After the projector is set, one can call ct_projector.fp()
        to use the projector. This will enable same reconstruction algorithm with different projectors.
        The set projector will be called as projector(ct_projector, image, **kwargs), where the image is
        an array, and it will return an array of projection.

        Parameters
        -----------------
        projector: callback function.
            The parameters must be projector(ct_projector, img, **kwargs). The first parameter is
            a ct_projector class, the second parameter is the image.
            The return value should be the forward projections in cp.array.
        **kwargs: parameters.
            kwargs to be parsed when calling the projector. For example, one can pass how many
            angles are there for the projector.
        '''
        self.projector = projector
        self.fp_kwargs = kwargs

    def set_backprojector(self, backprojector: Callable[..., cp.array], **kwargs) -> None:
        '''
        Set the backprojector. After the backprojector is set, one can call ct_projector.bp()
        to use the backprojector. This will enable same reconstruction algorithm with different projectors.
        The set projector will be called as backprojector(ct_projector, prj, **kwargs), where the prj is
        an array, and it will return an array of image.

        Parameters
        -----------------
        projector: callback function.
            The parameters must be backprojector(ct_projector, prj, **kwargs). The first parameter is
            a ct_projector class, the second parameter is the projection.
            The return value should be the back projections (an image) in array.
        **kwargs: parameters.
            kwargs to be parsed when calling the backprojector. For example, one can pass how many
            angles are there for the projector.
        '''
        self.backprojector = backprojector
        self.bp_kwargs = kwargs

    def fp(self, img: cp.array, **kwargs) -> cp.array:
        '''
        Generic forward projection function, the **kwargs will override
        the default params set by self.set_projector.

        Only need to set the parameters that is different from the default one.
        '''
        for k in self.fp_kwargs:
            if k not in kwargs:
                kwargs[k] = self.fp_kwargs[k]
        return self.projector(self, img, **kwargs)

    def bp(self, prj: cp.array, **kwargs) -> cp.array:
        '''
        Generic backprojection function, the **kwargs will override
        the default params set by self.set_backprojector.

        Only need to set the parameters that is different from the default one.
        '''
        for k in self.bp_kwargs:
            if k not in kwargs:
                kwargs[k] = self.bp_kwargs[k]
        return self.backprojector(self, prj, **kwargs)

    def calc_projector_norm(self, weight: cp.array = None, niter: int = 10) -> float:
        '''
        Use power method to calculate the norm of the projector.

        Parameters
        ----------------------
        weight: cp.array of size [1, self.nz, self.ny, self.nx]
            The weighting matrix as in A^TwA when calculating the norm of projector A.
        niter: int.
            Number of iterations for the power method.

        Returns
        --------------------
        norm: float.
            The norm of the projector A, or sqrt(w)A with weighting matrix.
        '''
        if weight is not None:
            weight = cp.sqrt(weight)
        else:
            weight = 1

        x = cp.random.uniform(size=[1, self.nz, self.ny, self.nx], dtype=cp.float32)
        x = x / cp.linalg.norm(x)

        for i in range(niter):
            print(i, end=',', flush=True)
            fp = self.fp(x)
            norm = cp.linalg.norm(fp)
            x = self.bp(fp * weight)

            x = x / cp.linalg.norm(x)
        print('')

        return norm

    def calc_norm_img(self, weight: cp.array = None) -> cp.array:
        '''
        Calculate norm_img = A.T*w*A*1

        Parameters
        --------------
        weight: cp.array of size [1, self.nz, self.ny, self.nx].
            The weighting matrix as in A^TwA.

        Returns
        -------------
        norm_img: cp.array.
            The calculated norm image.
        '''
        if weight is None:
            weight = 1

        x = cp.ones([1, self.nz, self.ny, self.nx], dtype=cp.float32)
        return self.bp(self.fp(x) * weight)

    def get_angles(self) -> np.array:
        '''
        Get the angles for each view in circular geometry.
        '''
        return np.arange(0, self.nview, dtype=np.float32) * 2 * np.pi / self.nview
