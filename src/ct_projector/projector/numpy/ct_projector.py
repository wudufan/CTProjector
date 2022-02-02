'''
Numpy wrapper among the cuda projectors. It is slower than the cupy version but
costs less GPU memory.
It has the same interface with the cupy version but some are not implemented.
'''

from ctypes import cdll, POINTER, c_float, c_int, c_ulong
from typing import Callable

import numpy as np
import configparser

import pkg_resources

module = cdll.LoadLibrary(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libprojector.so')
)


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

    def set_projector(self, projector: Callable[..., np.array], **kwargs) -> None:
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

    def set_backprojector(self, backprojector: Callable[..., np.array], **kwargs) -> None:
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

    def fp(self, img: np.array, **kwargs) -> np.array:
        '''
        Generic forward projection function, the **kwargs will override
        the default params set by self.set_projector.

        Only need to set the parameters that is different from the default one.
        '''
        for k in self.fp_kwargs:
            if k not in kwargs:
                kwargs[k] = self.fp_kwargs[k]
        return self.projector(img, **kwargs)

    def bp(self, prj: np.array, **kwargs) -> np.array:
        '''
        Generic backprojection function, the **kwargs will override
        the default params set by self.set_backprojector.

        Only need to set the parameters that is different from the default one.
        '''
        for k in self.bp_kwargs:
            if k not in kwargs:
                kwargs[k] = self.bp_kwargs[k]
        return self.backprojector(prj, **kwargs)

    def calc_projector_norm(self, weight: np.array = None, niter: int = 10) -> float:
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
            weight = np.sqrt(weight)
        else:
            weight = 1

        x = np.random.uniform(size=[1, self.nz, self.ny, self.nx], dtype=np.float32)
        x = x / np.linalg.norm(x)

        for i in range(niter):
            print(i, end=',', flush=True)
            fp = self.fp(x)
            norm = np.linalg.norm(fp)
            x = self.bp(fp * weight)

            x = x / np.linalg.norm(x)
        print('')

        return norm

    def calc_norm_img(self, weight: np.array = None) -> np.array:
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

        x = np.ones([1, self.nz, self.ny, self.nx], dtype=np.float32)
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

    def ramp_filter(self, prj: np.array, filter_type: str = 'hann') -> np.array:
        '''
        Filter the projection for equiangular geometry.

        Parameters
        ---------------
        prj: np.array(float32) of shape [batch, nview, nv, nu].
            The projection to be filtered. The shape will override the default ones,
            self.nview, self.nv, self.nu.
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

        module.cfbpFanFilter.restype = c_int

        err = module.cfbpFanFilter(
            fprj.ctypes.data_as(POINTER(c_float)),
            prj.ctypes.data_as(POINTER(c_float)),
            c_ulong(prj.shape[0]),
            c_ulong(prj.shape[3]),
            c_ulong(prj.shape[2]),
            c_ulong(prj.shape[1]),
            c_float(self.du / self.dsd),
            c_float(self.dv),
            c_float(self.off_u),
            c_float(self.off_v),
            c_float(self.dsd),
            c_float(self.dso),
            c_int(ifilter)
        )

        if err != 0:
            print(err)

        return fprj

    def fbp_fan_bp(self, prj: np.array, angles: np.array) -> np.array:
        '''
        Fanbeam backprojection with circular equiangular detector. Ray driven
        and weighted for FBP.

        Parameters
        ----------------
        prj: np.array(float32) of size [batch, nview, nv, nu].
            The projection to be backprojected. The size does not need to be the same
            with self.nview, self.nv, self.nu.
        angles: np.array(float32) of size [nview].
            The projection angles in radius.

        Returns
        --------------
        img: np.array(float32) of size [batch, self.nz, self.ny, self.nx]
            The backprojected image.
        '''

        prj = prj.astype(np.float32)
        angles = angles.astype(np.float32)
        img = np.zeros([prj.shape[0], self.nz, self.ny, self.nx], np.float32)

        module.cfbpFanBackprojection.restype = c_int

        err = module.cfbpFanBackprojection(
            img.ctypes.data_as(POINTER(c_float)),
            prj.ctypes.data_as(POINTER(c_float)),
            angles.ctypes.data_as(POINTER(c_float)),
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

        return img

    def siddon_fan_fp(self, img: np.array, angles: np.array) -> np.array:
        '''
        Fanbeam forward projection with circular equiangular detector. Siddon ray driven.

        Parameters
        ----------------
        img: np.array(float32) of size [batch, nz, ny, nx]
            The image to be projected.
        angles: np.array(float32) of size [nview]
            The projection angles in radius.

        Returns
        --------------
        prj: np.array(float32) of size [batch, self.nview, self.nv, self.nu]
            The forward projection.
        '''

        img = img.astype(np.float32)
        angles = angles.astype(np.float32)
        prj = np.zeros([img.shape[0], len(angles), self.nv, self.nu], np.float32)

        module.cSiddonFanProjection.restype = c_int

        err = module.cSiddonFanProjection(
            prj.ctypes.data_as(POINTER(c_float)),
            img.ctypes.data_as(POINTER(c_float)),
            angles.ctypes.data_as(POINTER(c_float)),
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

    def siddon_fan_bp(self, prj: np.array, angles: np.array) -> np.array:
        '''
        Fanbeam backprojection with circular equiangular detector. Siddon ray driven.

        Parameters
        ----------------
        prj: np.array(float32) of size [batch, nview, nv, nu].
            The projection to be backprojected. The size does not need to be the same
            with self.nview, self.nv, self.nu.
        angles: np.array(float32) of size [nview].
            The projection angles in radius.

        Returns
        --------------
        img: np.array(float32) of size [batch, self.nz, self.ny, self.nx]
            The backprojected image.
        '''

        prj = prj.astype(np.float32)
        angles = angles.astype(np.float32)
        img = np.zeros([prj.shape[0], self.nz, self.ny, self.nx], np.float32)

        module.cSiddonFanBackprojection.restype = c_int

        err = module.cSiddonFanBackprojection(
            img.ctypes.data_as(POINTER(c_float)),
            prj.ctypes.data_as(POINTER(c_float)),
            angles.ctypes.data_as(POINTER(c_float)),
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

        return img

    def siddon_cone_fp_arbitrary(
        self,
        img: np.array,
        det_center: np.array,
        det_u: np.array,
        det_v: np.array,
        src: np.array
    ) -> np.array:
        '''
        Conebeam forward projection with arbitrary geometry and flat panel. Using Siddon ray tracing.
        The size of the img will override that of self.nx, self.ny, self.nz. The projection size
        will be [batch, nview, self.nv, self.nu].

        Parameters
        -------------------------
        img: np.array(float32) of size [batch, nz, ny, nx].
            The image to be projected, nz, ny, nx can be different than self.nz, self.ny, self.nx.
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
        prj: np.array(float32) of size [batch, self.nview, self.nv, self.nu].
            The forward projection.
        '''

        # make sure they are float32
        img = img.astype(np.float32)
        det_center = det_center.astype(np.float32)
        det_u = det_u.astype(np.float32)
        det_v = det_v.astype(np.float32)
        src = src.astype(np.float32)

        # projection of size
        prj = np.zeros([img.shape[0], det_center.shape[0], self.nv, self.nu], np.float32)

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
        prj: np.array,
        det_center: np.array,
        det_u: np.array,
        det_v: np.array,
        src: np.array
    ) -> np.array:
        '''
        Conebeam backprojection with arbitrary geometry and flat panel. Using Siddon ray tracing.
        The size of the img will override that of self.nx, self.ny, self.nz. The projection size
        will be [batch, nview, self.nv, self.nu].

        Parameters
        -------------------------
        prj: np.array(float32) of size [batch, nview, nv, nu].
            The projection to be backprojected. It will override the default shape predefined,
            i.e. self.nview, self.nv, self.nu.
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
        img: np.array(float32) of size [batch, self.nz, self.ny, self.nx].
            The backprojected image.
        '''

        # make sure they are float32
        prj = prj.astype(np.float32)
        det_center = det_center.astype(np.float32)
        det_u = det_u.astype(np.float32)
        det_v = det_v.astype(np.float32)
        src = src.astype(np.float32)

        # projection of size
        img = np.zeros([prj.shape[0], self.nz, self.ny, self.nx], np.float32)

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
        img: np.array,
        det_center: np.array,
        src: np.array,
        branchless: bool = False
    ) -> np.array:
        '''
        Distance driven forward projection for tomosynthesis. It assumes that the detector has
        u=(1,0,0) and v = (0,1,0).
        The projection should be along the z-axis (main axis for distance driven projection).
        The img size will override the default predefined shape. The forward projection shape
        will be [batch, self.nview, self.nv, self.nu].

        Parameters
        -------------------
        img: np.array(float32) of size (batch, nz, ny, nx).
            The image to be projected, nz, ny, nx can be different than self.nz, self.ny, self.nx.
            The projector will always use the size of the image.
        det_center: np.array(float32) of size [nview, 3].
            The center of the detector in mm. Each row records the center of detector as (z, y, x).
        src: np.array(float32) of size [nview, 3].
            The src positions in mm. Each row records in the source position as (z, y, x).
        branchless: bool
            If True, use the branchless mode (double precision required).

        Returns
        -------------------------
        prj: np.array(float32) of size [batch, self.nview, self.nv, self.nu].
            The forward projection.
        '''
        img = img.astype(np.float32)
        det_center = det_center.astype(np.float32)
        src = src.astype(np.float32)

        prj = np.zeros([img.shape[0], det_center.shape[0], self.nv, self.nu], np.float32)

        if branchless:
            type_projector = 1
        else:
            type_projector = 0

        module.cDistanceDrivenTomoProjection.restype = c_int
        err = module.cDistanceDrivenTomoProjection(
            prj.ctypes.data_as(POINTER(c_float)),
            img.ctypes.data_as(POINTER(c_float)),
            det_center.ctypes.data_as(POINTER(c_float)),
            src.ctypes.data_as(POINTER(c_float)),
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
            c_float(self.off_v),
            c_int(type_projector)
        )

        if err != 0:
            print(err)

        return prj

    def distance_driven_bp_tomo(
        self,
        prj: np.array,
        det_center: np.array,
        src: np.array,
        branchless: bool = False
    ) -> np.array:
        '''
        Distance driven backprojection for tomosynthesis. It assumes that the detector has
        u=(1,0,0) and v = (0,1,0).
        The backprojection should be along the z-axis (main axis for distance driven projection).
        The size of the img will override that of self.nx, self.ny, self.nz. The projection size
        will be [batch, nview, self.nv, self.nu].

        Parameters
        -------------------------
        prj: np.array(float32) of size [batch, nview, nv, nu].
            The projection to be backprojected. It will override the default shape predefined,
            i.e. self.nview, self.nv, self.nu.
        det_center: np.array(float32) of size [nview, 3].
            The center of the detector in mm. Each row records the center of detector as (z, y, x).
        src: np.array(float32) of size [nview, 3].
            The src positions in mm. Each row records in the source position as (z, y, x).
        branchless: bool
            If True, use the branchless mode (double precision required).

        Returns
        -------------------------
        img: np.array(float32) of size [batch, self.nz, self.ny, self.nx].
            The backprojected image.
        '''

        prj = prj.astype(np.float32)
        det_center = det_center.astype(np.float32)
        src = src.astype(np.float32)

        img = np.zeros([prj.shape[0], self.nz, self.ny, self.nx], np.float32)

        if branchless:
            type_projector = 1
        else:
            type_projector = 0

        module.cDistanceDrivenTomoBackprojection.restype = c_int
        err = module.cDistanceDrivenTomoBackprojection(
            img.ctypes.data_as(POINTER(c_float)),
            prj.ctypes.data_as(POINTER(c_float)),
            det_center.ctypes.data_as(POINTER(c_float)),
            src.ctypes.data_as(POINTER(c_float)),
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
            c_float(self.off_v),
            c_int(type_projector)
        )

        if err != 0:
            print(err)

        return img
