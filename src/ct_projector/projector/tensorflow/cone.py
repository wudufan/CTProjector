'''
Conebeam geometry
'''

import numpy as np
import tensorflow as tf
from typing import List, Tuple, Any

from .ct_projector import ct_projector, tile_tensor

import pkg_resources

module = tf.load_op_library(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libtfprojector.so')
)


# %%
def make_geometry(
    det_center: np.array,
    det_u: np.array,
    det_v: np.array,
    src: np.array
) -> np.array:
    '''
    Make the numpy input to the geometry required for siddon_cone_fp(bp)_arbitrary.

    det_center: np.array(float32) of size [nview, 3].
        The center of the detector in mm. Each row records the center of detector as (z, y, x).
    det_u: np.array(float32) of size [nview, 3].
        The u axis of the detector. Each row is a normalized vector in (z, y, x).
    det_v: np.array(float32) of size [nview ,3].
        The v axis of the detector. Each row is a normalized vector in (z, y, x).
    src: np.array(float32) of size [nview, 3].
        The src positions in mm. Each row records in the source position as (z, y, x).
    '''

    return np.concatenate((det_center, det_u, det_v, src), 0).astype(np.float32)


# %%
def siddon_fp_arbitrary(
    projector: ct_projector,
    img: tf.Tensor,
    geometry: tf.Tensor,
    grid: tf.Tensor = None,
    detector: tf.Tensor = None,
    output_shape: tf.Tensor = None,
    default_shape: List[int] = None,
    name: str = None
) -> tf.Tensor:
    '''
    Conebeam forward projection with arbitrary geometry and flat panel. Using Siddon ray tracing.

    Parameters
    ----------------
    img: Tensor of shape [batch, nz, ny, nx, channel], float32.
        The image to be projected.
    geometry: Tensor of shape [batch, nview * 4, 3], float32.
        It is the stack of det_center, det_u, det_v, and src vectors.
    grid: Tensor of shape [batch, 6], float32.
        (dx, dy, dz, cx, cy, cz). If None, take the values from class attributes.
    detector: Tensor of shape [batch, 4], float32.
        (du, dv, off_u, off_v). If None, take the values from class attributes.
    output_shape: Tensor of shape [None, 3], int32.
        (nview, nv, nu). Only the first row will be taken for the output shape.
        If None, take the values from default_shape.
    default_shape: List of int with length 3.
        (nview, nv, nu). If None, take the values from class attributes.
    name: str.
        Name of the operation

    Returns
    ----------------
    prj: Tensor of shape [batch, nview, nv, nu, channel], float32.
        The forward projection.
    '''
    batchsize = tf.shape(img)[0]

    if grid is None:
        grid = np.array(
            [projector.dx, projector.dy, projector.dz, projector.cx, projector.cy, projector.cz], np.float32
        )
    if detector is None:
        detector = np.array([projector.du, projector.dv, projector.off_u, projector.off_v], np.float32)
    if default_shape is None:
        default_shape = [geometry.shape[1] // 4, projector.nv, projector.nu]
    if output_shape is None:
        output_shape = np.copy(default_shape).astype(np.int32)

    grid = tile_tensor(grid, batchsize)
    detector = tile_tensor(detector, batchsize)
    output_shape = tile_tensor(output_shape, batchsize)

    # (batch, channel, nz, ny, nx)
    img = tf.transpose(img, (0, 4, 1, 2, 3))

    prj = module.siddon_cone_fp(
        image=img,
        geometry=geometry,
        grid=grid,
        detector=detector,
        output_shape=output_shape,
        default_shape=default_shape,
        name=name
    )

    # reshape it back
    # (batch, nu, nv, nview, channel)
    prj = tf.transpose(prj, (0, 2, 3, 4, 1))

    return prj


def siddon_bp_arbitrary(
    projector: ct_projector,
    prj: tf.Tensor,
    geometry: tf.Tensor,
    grid: tf.Tensor = None,
    detector: tf.Tensor = None,
    output_shape: tf.Tensor = None,
    default_shape: List[int] = None,
    name: str = None
):
    '''
    Conebeam backprojection with arbitrary geometry and flat panel. Using Siddon ray tracing.
    Parameters
    ----------------
    prj: Tensor of shape [batch, nview, nv, nu, channel], float32.
        The projection to be backprojected.
    geometry: Tensor of shape [batch, nview * 4, 3], float32.
        It is the stack of det_center, det_u, det_v, and src vectors.
    grid: Tensor of shape [batch, 6], float32.
        (dx, dy, dz, cx, cy, cz). If None, take the values from class attributes.
    detector: Tensor of shape [batch, 4], float32.
        (du, dv, off_u, off_v). If None, take the values from class attributes.
    output_shape: Tensor of shape [None, 3], int32.
        (nview, nv, nu). Only the first row will be taken for the output shape.
        If None, take the values from default_shape.
    default_shape: List of int with length 3.
        (nview, nv, nu). If None, take the values from class attributes.
    name: str.
        Name of the operation

    Returns
    ----------------
    img: Tensor of shape [batch, nz, ny, nx, channel], float32.
        The backprojected image.
    '''
    batchsize = tf.shape(prj)[0]

    if grid is None:
        grid = np.array(
            [projector.dx, projector.dy, projector.dz, projector.cx, projector.cy, projector.cz], np.float32
        )
    if detector is None:
        detector = np.array([projector.du, projector.dv, projector.off_u, projector.off_v], np.float32)
    if default_shape is None:
        default_shape = [projector.nx, projector.ny, projector.nz]
    if output_shape is None:
        output_shape = np.copy(default_shape).astype(np.int32)

    grid = tile_tensor(grid, batchsize)
    detector = tile_tensor(detector, batchsize)
    output_shape = tile_tensor(output_shape, batchsize)

    # reshape the img tensor for input
    # (batch, channel, nview, nv, nu)
    prj = tf.transpose(prj, (0, 4, 1, 2, 3))

    img = module.siddon_cone_bp(
        projection=prj,
        geometry=geometry,
        grid=grid,
        detector=detector,
        output_shape=output_shape,
        default_shape=default_shape,
        name=name
    )

    # reshape it back
    # (batch, nx, ny, nz, channel)
    img = tf.transpose(img, (0, 2, 3, 4, 1))

    return img


# register gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


@ops.RegisterGradient("SiddonConeFP")
def _siddon_cone_fp_grad(
    op: tf.Operation, grad: tf.Tensor
) -> Tuple[tf.Tensor, Any, Any, Any, Any]:
    '''
    The gradients for SiddonConeFP

    Parameters
    ------------------
    op: tf.Operation.
        The SiddonConeFP operator. See relevant class for more information.
    grad: tf.Tensor.
        The gradient w.r.t. output of SiddonConeFP

    Returns
    ------------------
    Tuple of length 5.
        The first element is the gradient w.r.t. input image of SiddonConeFP.
        Others will be None.
    '''
    # get shape
    input_shape = array_ops.shape(op.inputs[0])
    batchsize = input_shape[0]
    input_shape = tf.tile(input_shape[2:][tf.newaxis], [batchsize, 1])

    # other params
    geometry = op.inputs[1]
    grid = op.inputs[2]
    detector = op.inputs[3]

    # backprojection
    bp = module.siddon_cone_bp(
        projection=grad,
        geometry=geometry,
        grid=grid,
        detector=detector,
        output_shape=input_shape,
        default_shape=op.inputs[0].shape[2:].as_list()
    )

    return [bp, None, None, None, None]


@ops.RegisterGradient("SiddonConeBP")
def _siddon_cone_bp_grad(
    op: tf.Operation, grad: tf.Tensor
) -> Tuple[tf.Tensor, Any, Any, Any, Any]:
    '''
    The gradients for SiddonConeBP

    Parameters
    ------------------
    op: tf.Operation.
        The SiddonConeBP operator. See relevant class for more information.
    grad: tf.Tensor.
        The gradient w.r.t. output of SiddonConeBP

    Returns
    ------------------
    Tuple of length 5.
        The first element is the gradient w.r.t. input projection of SiddonConeBP.
        Others will be None.
    '''
    # get shape
    input_shape = array_ops.shape(op.inputs[0])
    batchsize = input_shape[0]
    input_shape = tf.tile(input_shape[2:][tf.newaxis], [batchsize, 1])

    # other params
    geometry = op.inputs[1]
    grid = op.inputs[2]
    detector = op.inputs[3]

    # forward projection
    fp = module.siddon_cone_fp(
        image=grad,
        geometry=geometry,
        grid=grid,
        detector=detector,
        output_shape=input_shape,
        default_shape=op.inputs[0].shape[2:].as_list()
    )

    return [fp, None, None, None, None]


# keras modules
class SiddonFPArbitrary(tf.keras.layers.Layer):
    '''
    The keras module for forward projection.

    Initialization Parameters
    --------------------
    projector: ct_projector class.
        It holds the default geometries.
    default_shape: list of int with length 3.
        It is corresponding to the forward projection shape in [nview, nv, nu].
        Although the output shape can be passed during computation via output_shape, default_shape provides
        shape inference capability. Any element of default_shape can be -1, which means that the corresponding
        dimension will be fetched from output_shape during computation. The default_shape can also be None,
        then the nview will be derived from the geometry input, and nv/nu will be taken from projector.

    Call Parameters
    ------------------
    Mandatory:
    inputs: Tensor of shape [batch, nz, ny, nx, channel], float32.
        It is the image to be forward projected.
    geometry: Tensor of shape [batch, nview * 4, 3], float 32.
        Different geometries can be set to different elements in one batch.
        Geometry should be passed by kwargs only, i.e. SiddonConeFP(...)(..., geometry = geometry).
        For each elements (geometry[i]), it consists of the following 4 part, each with size [nview, 3]:
        det_centers: the central position of the detector. (x, y, z) at each view;
        det_us: the axis of the U direction of the detector in the world coordinates.
            (x, y, z) at each view. It should be normalized.
        det_vs: the axis of the V direction of the detector in the world coordinates.
            (x, y, z) at each view. It should be normalized and perpendicular to det_us.
        srcs: the source position, (x, y, z) at each view.

    Optional:
    grid: Tensor of shape [batch, 6] or [6], float32.
        Each elements corresponding to (dx, dy, dz, cx, cy, cz) in mm.
        If it is in [batch, 6], each row sets a different grid for the corresponding input image.
        If it is [6], it sets the same grid for all the input images.
        If it is not presented or None, will take value from self.projector.
    detector: Tensor of shape [batch, 4] or [4], float32.
        Each elements corresponding to (du, dv, off_u, off_v). du/dv are in mm and off_u/off_v are in pixels.
        If it is in [batch, 4], each row sets a different detector for the corresponding output projection.
        If it is [4], it sets the same detector for all the output pojections.
        If it is not presented or None, will take value from self.projector.
    output_shape: Tensor of shape [None, 3] or [3], float32.
        Each elements corresponding to (nview, nv, nu).
        Only the first row in output_shape will be used as the output shape.
        If it is not presented or None, will take value from default_shape.

    Returns
    ------------------
    prj: Tensor of shape [batch, nview, nv, nu, channel], float32.
        The forward projection.

    '''
    def __init__(
        self,
        projector: ct_projector,
        default_shape: List[int] = None,
        name: str = ''
    ):
        super(SiddonFPArbitrary, self).__init__(name=name)
        self.projector = projector
        self.default_shape = default_shape

    def build(self, input_shape):
        super(SiddonFPArbitrary, self).build(input_shape)

    def call(self, inputs, **kwargs):
        geometry = kwargs['geometry']
        if 'grid' in kwargs:
            grid = kwargs['grid']
        else:
            grid = None
        if 'detector' in kwargs:
            detector = kwargs['detector']
        else:
            detector = None
        if 'output_shape' in kwargs:
            output_shape = kwargs['output_shape']
        else:
            output_shape = None

        return siddon_fp_arbitrary(
            self.projector, inputs, geometry, grid, detector, output_shape, self.default_shape
        )


class SiddonBPArbitrary(tf.keras.layers.Layer):
    '''
    Similar parameters to SiddonConeFP.
    '''
    def __init__(
        self,
        projector: ct_projector,
        default_shape: List[int] = None,
        name: str = ''
    ):
        super(SiddonBPArbitrary, self).__init__(name=name)
        self.projector = projector
        self.default_shape = default_shape

    def build(self, input_shape):
        super(SiddonBPArbitrary, self).build(input_shape)

    def call(self, inputs, **kwargs):
        geometry = kwargs['geometry']
        if 'grid' in kwargs:
            grid = kwargs['grid']
        else:
            grid = None
        if 'detector' in kwargs:
            detector = kwargs['detector']
        else:
            detector = None
        if 'output_shape' in kwargs:
            output_shape = kwargs['output_shape']
        else:
            output_shape = None

        return siddon_bp_arbitrary(
            self.projector, inputs, geometry, grid, detector, output_shape, self.default_shape
        )
