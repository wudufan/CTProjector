'''
Cupy-based wrapper of the CUDA functions. Compared to numpy wrapper, it reduces memory transfer
between CPU and GPU but costs more GPU memory.
'''

from ctypes import cdll, c_int, c_void_p, c_ulong, c_float
from typing import Callable, Union

import os
import cupy as cp
import numpy as np
import configparser

module = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libprojector.so'))


class ct_projector:
    '''
    CT projector wrapper
    '''
    def __init__(self):
        self.nview = 720
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
        or ct_projector.fp2() to use the projector. This will enable same reconstruction
        algorithm with different projectors.
        The set projector will be called as projector(image, **kwargs), where the image is
        an array, and it will return an array of projection.

        Parameters
        -----------------
        projector: callback function.
            It takes the first parameter an image array, and the rest parameters given by **kwargs.
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
        or ct_projector.bp2() to use the backprojector. This will enable same reconstruction
        algorithm with different projectors.
        The set projector will be called as backprojector(prj, **kwargs), where the prj is
        an array, and it will return an array of image.

        Parameters
        -----------------
        projector: callback function.
            It takes the first parameter a projection array, and the rest parameters given by **kwargs.
            The return value should be the back projections (an image) in array.
        **kwargs: parameters.
            kwargs to be parsed when calling the backprojector. For example, one can pass how many
            angles are there for the projector.
        '''
        self.backprojector = backprojector
        self.bp_kwargs = kwargs

    def fp(self, img: cp.array) -> cp.array:
        '''
        Generic forward projection function.
        '''
        return self.projector(img, **(self.fp_kwargs))

    def bp(self, prj: cp.array) -> cp.array:
        '''
        Generic backprojection function.
        '''
        return self.backprojector(prj, **(self.bp_kwargs))

    def fp2(self, img: cp.array, **kwargs) -> cp.array:
        '''
        Generic forward projection function, the **kwargs will override
        the default params set by self.set_projector.
        '''
        return self.projector(img, **kwargs)

    def bp2(self, prj: cp.array, **kwargs) -> cp.array:
        '''
        Generic backprojection function, the **kwargs will override
        the default params set by self.set_backprojector.
        '''
        return self.backprojector(prj, **kwargs)

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

    def set_device(self, device: int) -> int:
        '''
        Set the computing device and return any error code.
        '''
        return module.SetDevice(c_int(device))

    def get_angles(self) -> np.array:
        '''
        Get the angles for each view in circular geometry.
        '''
        return np.arange(0, self.nview, dtype=np.float32) * 2 * np.pi / self.nview

    def siddon_cone_fp_arbitrary(
        self,
        img: cp.array,
        det_center: cp.array,
        det_u: cp.array,
        det_v: cp.array,
        src: cp.array
    ) -> cp.array:
        '''
        Conebeam forward projection with arbitrary geometry and flat panel. Using Siddon ray tracing.
        The size of the img will override that of self.nx, self.ny, self.nz. The projection size
        will be [batch, nview, self.nv, self.nu].

        Parameters
        -------------------------
        img: cp.array(float32) of size [batch, nz, ny, nx].
            The image to be projected, nz, ny, nx can be different than self.nz, self.ny, self.nx.
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
        prj: cp.array(float32) of size [batch, self.nview, self.nv, self.nu].
            The forward projection.
        '''

        # projection of size
        prj = cp.zeros([img.shape[0], det_center.shape[0], self.nv, self.nu], cp.float32)

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
            c_float(self.dx),
            c_float(self.dy),
            c_float(self.dz),
            c_float(self.cx),
            c_float(self.cy),
            c_float(self.cz),
            c_ulong(prj.shape[3]),
            c_ulong(prj.shape[2]),
            c_ulong(prj.shape[1]),
            c_float(self.du),
            c_float(self.dv),
            c_float(self.off_u),
            c_float(self.off_v)
        )

        if err != 0:
            print(err)

        return prj

    def siddon_cone_bp_arbitrary(
        self,
        prj: cp.array,
        det_center: cp.array,
        det_u: cp.array,
        det_v: cp.array,
        src: cp.array
    ) -> cp.array:
        '''
        Conebeam backprojection with arbitrary geometry and flat panel. Using Siddon ray tracing.
        The size of the img will override that of self.nx, self.ny, self.nz. The projection size
        will be [batch, nview, self.nv, self.nu].

        Parameters
        -------------------------
        prj: cp.array(float32) of size [batch, nview, nv, nu].
            The projection to be backprojected. It will override the default shape predefined,
            i.e. self.nview, self.nv, self.nu.
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
        img: cp.array(float32) of size [batch, self.nz, self.ny, self.nx].
            The backprojected image.
        '''

        # make sure they are float32
        # projection of size
        img = cp.zeros([prj.shape[0], self.nz, self.ny, self.nx], cp.float32)

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
            c_float(self.dx),
            c_float(self.dy),
            c_float(self.dz),
            c_float(self.cx),
            c_float(self.cy),
            c_float(self.cz),
            c_ulong(prj.shape[3]),
            c_ulong(prj.shape[2]),
            c_ulong(prj.shape[1]),
            c_float(self.du),
            c_float(self.dv),
            c_float(self.off_u),
            c_float(self.off_v)
        )

        if err != 0:
            print(err)

        return img

    def distance_driven_fp_tomo(
        self,
        img: cp.array,
        det_center: cp.array,
        src: cp.array
    ) -> cp.array:
        '''
        Distance driven forward projection for tomosynthesis. It assumes that the detector has
        u=(1,0,0) and v = (0,1,0).
        The projection should be along the z-axis (main axis for distance driven projection).
        The img size will override the default predefined shape. The forward projection shape
        will be [batch, self.nview, self.nv, self.nu].

        Parameters
        -------------------
        img: cp.array(float32) of size (batch, nz, ny, nx).
            The image to be projected, nz, ny, nx can be different than self.nz, self.ny, self.nx.
            The projector will always use the size of the image.
        det_center: cp.array(float32) of size [nview, 3].
            The center of the detector in mm. Each row records the center of detector as (z, y, x).
        src: cp.array(float32) of size [nview, 3].
            The src positions in mm. Each row records in the source position as (z, y, x).

        Returns
        -------------------------
        prj: cp.array(float32) of size [batch, self.nview, self.nv, self.nu].
            The forward projection.
        '''
        prj = cp.zeros([img.shape[0], det_center.shape[0], self.nv, self.nu], cp.float32)

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
            c_float(self.dx),
            c_float(self.dy),
            c_float(self.dz),
            c_float(self.cx),
            c_float(self.cy),
            c_float(self.cz),
            c_ulong(prj.shape[3]),
            c_ulong(prj.shape[2]),
            c_ulong(prj.shape[1]),
            c_float(self.du),
            c_float(self.dv),
            c_float(self.off_u),
            c_float(self.off_v)
        )

        if err != 0:
            print(err)

        return prj

    def distance_driven_bp_tomo(
        self,
        prj: cp.array,
        det_center: cp.array,
        src: cp.array
    ) -> cp.array:
        '''
        Distance driven backprojection for tomosynthesis. It assumes that the detector has
        u=(1,0,0) and v = (0,1,0).
        The backprojection should be along the z-axis (main axis for distance driven projection).
        The size of the img will override that of self.nx, self.ny, self.nz. The projection size
        will be [batch, nview, self.nv, self.nu].

        Parameters
        -------------------------
        prj: cp.array(float32) of size [batch, nview, nv, nu].
            The projection to be backprojected. It will override the default shape predefined,
            i.e. self.nview, self.nv, self.nu.
        det_center: cp.array(float32) of size [nview, 3].
            The center of the detector in mm. Each row records the center of detector as (z, y, x).
        src: cp.array(float32) of size [nview, 3].
            The src positions in mm. Each row records in the source position as (z, y, x).

        Returns
        -------------------------
        img: cp.array(float32) of size [batch, self.nz, self.ny, self.nx].
            The backprojected image.
        '''

        img = cp.zeros([prj.shape[0], self.nz, self.ny, self.nx], cp.float32)

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
            c_float(self.dx),
            c_float(self.dy),
            c_float(self.dz),
            c_float(self.cx),
            c_float(self.cy),
            c_float(self.cz),
            c_ulong(prj.shape[3]),
            c_ulong(prj.shape[2]),
            c_ulong(prj.shape[1]),
            c_float(self.du),
            c_float(self.dv),
            c_float(self.off_u),
            c_float(self.off_v)
        )

        if err != 0:
            print(err)

        return img

    def distance_driven_fan_fp(
        self,
        img: cp.array,
        angles: cp.array
    ) -> cp.array:
        '''
        Fanbeam forward projection with circular equiangular detector. Distance driven.

        Parameters
        ----------------
        img: cp.array(float32) of size [batch, nz, ny, nx]
            The image to be projected.
        angles: cp.array(float32) of size [nview]
            The projection angles in radius.

        Returns
        --------------
        prj: cp.array(float32) of size [batch, self.nview, self.nv, self.nu]
            The forward projection.
        '''
        prj = cp.zeros([img.shape[0], len(angles), self.nv, self.nu], cp.float32)

        module.cupyDistanceDrivenFanProjection.restype = c_int

        err = module.cupyDistanceDrivenFanProjection(
            c_void_p(prj.data.ptr),
            c_void_p(img.data.ptr),
            c_void_p(angles.data.ptr),
            c_ulong(img.shape[0]),
            c_ulong(img.shape[3]),
            c_ulong(img.shape[2]),
            c_ulong(img.shape[1]),
            c_float(self.dx),
            c_float(self.dy),
            c_float(self.dz),
            c_float(self.cx),
            c_float(self.cy),
            c_float(self.cz),
            c_ulong(prj.shape[3]),
            c_ulong(prj.shape[2]),
            c_ulong(prj.shape[1]),
            c_float(self.du / self.dsd),
            c_float(self.dv),
            c_float(self.off_u),
            c_float(self.off_v),
            c_float(self.dsd),
            c_float(self.dso)
        )

        if err != 0:
            print(err)

        return prj

    def distance_driven_fan_bp(
        self,
        prj: cp.array,
        angles: cp.array,
        is_fbp: Union[bool, int] = False
    ) -> cp.array:
        '''
        Fanbeam backprojection with circular equiangular detector. Distance driven.

        Parameters
        ----------------
        prj: cp.array(float32) of size [batch, nview, nv, nu].
            The projection to be backprojected. The size does not need to be the same
            with self.nview, self.nv, self.nu.
        angles: cp.array(float32) of size [nview].
            The projection angles in radius.
        is_fbp: bool.
            if true, use the FBP weighting scheme to backproject filtered data.

        Returns
        --------------
        img: cp.array(float32) of size [batch, self.nz, self.ny, self.nx]
            The backprojected image.
        '''

        img = cp.zeros([prj.shape[0], self.nz, self.ny, self.nx], cp.float32)
        if is_fbp:
            type_projector = 1
        else:
            type_projector = 0

        module.cupyDistanceDrivenFanBackprojection.restype = c_int

        err = module.cupyDistanceDrivenFanBackprojection(
            c_void_p(img.data.ptr),
            c_void_p(prj.data.ptr),
            c_void_p(angles.data.ptr),
            c_ulong(img.shape[0]),
            c_ulong(img.shape[3]),
            c_ulong(img.shape[2]),
            c_ulong(img.shape[1]),
            c_float(self.dx),
            c_float(self.dy),
            c_float(self.dz),
            c_float(self.cx),
            c_float(self.cy),
            c_float(self.cz),
            c_ulong(prj.shape[3]),
            c_ulong(prj.shape[2]),
            c_ulong(prj.shape[1]),
            c_float(self.du / self.dsd),
            c_float(self.dv),
            c_float(self.off_u),
            c_float(self.off_v),
            c_float(self.dsd),
            c_float(self.dso),
            c_int(type_projector)
        )

        if err != 0:
            print(err)

        return img

    def filter_tomo(
        self,
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
            i.e. self.nview, self.nv, self.nu.
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
            c_float(self.du),
            c_float(self.dx),
            c_float(self.dz),
            c_int(ifilter),
            c_float(cutoff_x),
            c_float(cutoff_y)
        )

        if err != 0:
            print(err)

        return fprj
