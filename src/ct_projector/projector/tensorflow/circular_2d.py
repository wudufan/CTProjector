'''
Circular 2D geometry
'''

# %%
import numpy as np
import tensorflow as tf
import inspect
from enum import IntEnum
from typing import List

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from .ct_projector import ct_projector, tile_tensor

import pkg_resources

module = tf.load_op_library(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libtfprojector.so')
)


# %%
class TypeGeometry(IntEnum):
    PARALLEL = 0
    FAN_EQUIANGULAR = 1


class TypeProjector(IntEnum):
    IR = 0
    FORCE_FBP = 4


# %%
def distance_driven_2d_fp(
    projector: ct_projector,
    img: tf.Tensor,
    angles: tf.Tensor,
    dsd: tf.Tensor = None,
    dso: tf.Tensor = None,
    grid: tf.Tensor = None,
    detector: tf.Tensor = None,
    output_shape: tf.Tensor = None,
    default_shape: List[int] = None,
    type_geometry: TypeGeometry = TypeGeometry.PARALLEL,
    type_projector: int = TypeProjector.IR,
    name: str = None
) -> tf.Tensor:
    '''
    2D forward projection, circular trajectory with distance driven

    Parameters
    ----------------
    img: Tensor of shape [batch, nz, ny, nx, channel], float32.
        The image to be projected.
    angles: Tensor of shape [batch, nview], float32.
        The projection angles in radius.
    dsd: Tensor of shape [batch, 1], float32.
        The distance from source to detector. If None, take the values from projector
    dso: Tensor of shape [batch, 1], float32.
        The distance from source to isocenter. If None, take the values from projector
    grid: Tensor of shape [batch, 6], float32.
        (dx, dy, dz, cx, cy, cz). If None, take the values from projector.
    detector: Tensor of shape [batch, 4], float32.
        (du, dv, off_u, off_v). If None, take the values from projector.
    output_shape: Tensor of shape [None, 3], int32.
        (nview, nv, nu). Only the first row will be taken for the output shape.
        If None, take the values from default_shape.
    default_shape: List of int with length 3.
        (nview, nv, nu). If None, take the values from projector.
    type_geometry: TypeGeometry.
        The geometry of the projection.
    type_projector: TypeProjector.
        IR: where the ray intersection length is considered.
        FORCE_FBP: the ray intersection length is ignored. It is useful when calculating the conjugate to
            the backprojection in FBP.
    name: str.
        Name of the operation

    Returns
    ----------------
    prj: Tensor of shape [batch, nview, nv, nu, channel], float32.
        The forward projection.
    '''
    batchsize = tf.shape(img)[0]

    angles = tile_tensor(angles, batchsize)

    if dsd is None:
        dsd = tf.ones([batchsize, 1]) * projector.dsd
    if dso is None:
        dso = tf.ones([batchsize, 1]) * projector.dso
    if grid is None:
        grid = np.array(
            [projector.dx, projector.dy, projector.dz, projector.cx, projector.cy, projector.cz], np.float32
        )
    if detector is None:
        if type_geometry == TypeGeometry.FAN_EQUIANGULAR:
            detector = np.array(
                [projector.du / projector.dsd, projector.dv, projector.off_u, projector.off_v], np.float32
            )
        else:
            detector = np.array(
                [projector.du, projector.dv, projector.off_u, projector.off_v], np.float32
            )
    if default_shape is None:
        default_shape = [angles.shape[1], projector.nv, projector.nu]
    if output_shape is None:
        output_shape = np.copy(default_shape).astype(np.int32)

    grid = tile_tensor(grid, batchsize)
    detector = tile_tensor(detector, batchsize)
    output_shape = tile_tensor(output_shape, batchsize)

    # (batch, channel, nz, ny, nx)
    img = tf.transpose(img, (0, 4, 1, 2, 3))

    prj = module.distance_driven_2d_fp(
        image=img,
        angles=angles,
        dsd=dsd,
        dso=dso,
        grid=grid,
        detector=detector,
        output_shape=output_shape,
        default_shape=default_shape,
        type_geometry=int(type_geometry),
        type_projector=type_projector,
        name=name
    )

    # reshape it back
    # (batch, nu, nv, nview, channel)
    prj = tf.transpose(prj, (0, 2, 3, 4, 1))

    return prj


@ops.RegisterGradient("DistanceDriven_2D_FP")
def _distance_driven_2d_fp_grad(
    op: tf.Operation, grad: tf.Tensor
):
    '''
    The gradients for SiddonConeFP

    Parameters
    ------------------
    op: tf.Operation.
        The DistanceDriven_2D_FP operator. See relevant class for more information.
    grad: tf.Tensor.
        The gradient w.r.t. output of DistanceDriven_2D_FP

    Returns
    ------------------
    Tuple of length 7.
        The first element is the gradient w.r.t. input image of DistanceDriven_2D_FP.
        Others will be None.
    '''
    # get shape
    # input_shape is (batch, channel, nz, ny, nx)
    # input_shape[2:] is (nz, ny, nx)
    # the output is a [nbatch, 3], array, where each row is (nz, ny, nx)
    input_shape = array_ops.shape(op.inputs[0])
    batchsize = input_shape[0]
    input_shape = tf.tile(input_shape[2:][tf.newaxis], [batchsize, 1])

    # other params
    type_geometry = op.get_attr('type_geometry')
    type_projector = op.get_attr('type_projector')
    angles = op.inputs[1]
    dso = op.inputs[2]
    dsd = op.inputs[3]
    grid = op.inputs[4]
    detector = op.inputs[5]

    # backprojection
    bp = module.distance_driven_2d_bp(
        projection=grad,
        angles=angles,
        dso=dso,
        dsd=dsd,
        grid=grid,
        detector=detector,
        output_shape=input_shape,
        type_geometry=type_geometry,
        type_projector=type_projector,
        default_shape=op.inputs[0].shape[2:].as_list()
    )

    return [bp, None, None, None, None, None, None]


class DistanceDriven2DFP(tf.keras.layers.Layer):
    '''
    The keras module for forward projection.

    Initialization Parameters
    --------------------
    projector: ct_projector class.
        It holds the default geometries.
    angles: Tensor of shape [batch, nview], float32.
        Default projection angles in radius. Can be overloaded during call().
    type_geometry: TypeGeometry.
        Default geometry of the projection. Can be overloaded during call().
    type_projector: TypeProjector.
        Default projector. Can be overloaded during call().
        IR: where the ray intersection length is considered.
        FORCE_FBP: the ray intersection length is ignored. It is useful when calculating the conjugate to
            the backprojection in FBP.
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

    Optional:
    angles: Tensor of shape [batch, nview], float32.
        The projection angles in radius.
    dsd: Tensor of shape [batch, 1], float32.
        The distance from source to detector. If None, take the values from projector
    dso: Tensor of shape [batch, 1], float32.
        The distance from source to isocenter. If None, take the values from projector
    grid: Tensor of shape [batch, 6], float32.
        (dx, dy, dz, cx, cy, cz). If None, take the values from projector.
    detector: Tensor of shape [batch, 4], float32.
        (du, dv, off_u, off_v). If None, take the values from projector.
    output_shape: Tensor of shape [None, 3], int32.
        (nview, nv, nu). Only the first row will be taken for the output shape.
        If None, take the values from default_shape.
    default_shape: List of int with length 3.
        (nview, nv, nu). If None, take the values from projector.

    Returns
    ------------------
    prj: Tensor of shape [batch, nview, nv, nu, channel], float32.
        The forward projection.

    '''
    def __init__(
        self,
        projector: ct_projector,
        angles: tf.Tensor = None,
        type_geometry: TypeGeometry = TypeGeometry.PARALLEL,
        type_projector: int = TypeProjector.IR,
        default_shape: List[int] = None,
        name: str = ''
    ):
        super(DistanceDriven2DFP, self).__init__(name=name)
        self.projector = projector
        self.angles = angles
        self.type_geometry = type_geometry
        self.type_projector = type_projector
        self.default_shape = default_shape

    def build(self, input_shape):
        super(DistanceDriven2DFP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        kwargs_list = inspect.getfullargspec(distance_driven_2d_fp)[0]
        kwargs_call = {
            'angles': self.angles,
            'type_geometry': self.type_geometry,
            'type_projector': self.type_projector
        }
        for name in kwargs_list:
            if name in ['projector', 'img', 'default_shape', 'name']:
                continue
            if name in kwargs:
                kwargs_call[name] = kwargs[name]

        return distance_driven_2d_fp(
            projector=self.projector,
            img=inputs,
            default_shape=self.default_shape,
            **kwargs_call
        )


# %%
def distance_driven_2d_bp(
    projector: ct_projector,
    prj: tf.Tensor,
    angles: tf.Tensor,
    dsd: tf.Tensor = None,
    dso: tf.Tensor = None,
    grid: tf.Tensor = None,
    detector: tf.Tensor = None,
    output_shape: tf.Tensor = None,
    default_shape: List[int] = None,
    type_geometry: TypeGeometry = TypeGeometry.PARALLEL,
    type_projector: int = TypeProjector.IR,
    name: str = None
) -> tf.Tensor:
    '''
    2D backprojection, circular trajectory with distance driven

    Parameters
    ----------------
    prj: Tensor of shape [batch, nview, nv, nu, channel], float32.
        The projection to be backprojected.
    angles: Tensor of shape [batch, nview], float32.
        The projection angles in radius.
    dsd: Tensor of shape [batch, 1], float32.
        The distance from source to detector. If None, take the values from projector
    dso: Tensor of shape [batch, 1], float32.
        The distance from source to isocenter. If None, take the values from projector
    grid: Tensor of shape [batch, 6], float32.
        (dx, dy, dz, cx, cy, cz). If None, take the values from projector.
    detector: Tensor of shape [batch, 4], float32.
        (du, dv, off_u, off_v). If None, take the values from projector.
    output_shape: Tensor of shape [None, 3], int32.
        (nview, nv, nu). Only the first row will be taken for the output shape.
        If None, take the values from default_shape.
    default_shape: List of int with length 3.
        (nview, nv, nu). If None, take the values from projector.
    type_geometry: TypeGeometry.
        The geometry of the projection.
    type_projector: TypeProjector.
        IR: where the ray intersection length is considered.
        FORCE_FBP: the ray intersection length is ignored. It is useful when calculating the conjugate to
            the backprojection in FBP.
    name: str.
        Name of the operation

    Returns
    ----------------
    img: Tensor of shape [batch, nz, ny, nx, channel], float32.
        The backprojected image.
    '''
    batchsize = tf.shape(prj)[0]

    angles = tile_tensor(angles, batchsize)

    if dsd is None:
        dsd = tf.ones([batchsize, 1]) * projector.dsd
    if dso is None:
        dso = tf.ones([batchsize, 1]) * projector.dso
    if grid is None:
        grid = np.array(
            [projector.dx, projector.dy, projector.dz, projector.cx, projector.cy, projector.cz], np.float32
        )
    if detector is None:
        if type_geometry == TypeGeometry.FAN_EQUIANGULAR:
            detector = np.array(
                [projector.du / projector.dsd, projector.dv, projector.off_u, projector.off_v], np.float32
            )
        else:
            detector = np.array(
                [projector.du, projector.dv, projector.off_u, projector.off_v], np.float32
            )
    if default_shape is None:
        default_shape = [projector.nz, projector.ny, projector.nx]
    if output_shape is None:
        output_shape = np.copy(default_shape).astype(np.int32)

    grid = tile_tensor(grid, batchsize)
    detector = tile_tensor(detector, batchsize)
    output_shape = tile_tensor(output_shape, batchsize)

    # (batch, channel, nview, nv, nu)
    prj = tf.transpose(prj, (0, 4, 1, 2, 3))

    img = module.distance_driven_2d_bp(
        projection=prj,
        angles=angles,
        dsd=dsd,
        dso=dso,
        grid=grid,
        detector=detector,
        output_shape=output_shape,
        default_shape=default_shape,
        type_geometry=int(type_geometry),
        type_projector=type_projector,
        name=name
    )

    # reshape it back
    # (batch, nx, ny, nz, channel)
    img = tf.transpose(img, (0, 2, 3, 4, 1))

    return img


@ops.RegisterGradient("DistanceDriven_2D_BP")
def _distance_driven_2d_bp_grad(
    op: tf.Operation, grad: tf.Tensor
):
    '''
    The gradients for SiddonConeFP

    Parameters
    ------------------
    op: tf.Operation.
        The DistanceDriven_2D_BP operator. See relevant class for more information.
    grad: tf.Tensor.
        The gradient w.r.t. output of DistanceDriven_2D_BP

    Returns
    ------------------
    Tuple of length 7.
        The first element is the gradient w.r.t. input projection of DistanceDriven_2D_BP.
        Others will be None.
    '''
    # get shape
    # input_shape is (batch, channel, nview, nv, nu)
    # input_shape[2:] is (nview, nv, nu)
    # the output is a [nbatch, 3], array, where each row is (nview, nv, nu)
    input_shape = array_ops.shape(op.inputs[0])
    batchsize = input_shape[0]
    input_shape = tf.tile(input_shape[2:][tf.newaxis], [batchsize, 1])

    # other params
    type_geometry = op.get_attr('type_geometry')
    type_projector = op.get_attr('type_projector')
    angles = op.inputs[1]
    dso = op.inputs[2]
    dsd = op.inputs[3]
    grid = op.inputs[4]
    detector = op.inputs[5]

    # forward projection
    fp = module.distance_driven_2d_fp(
        image=grad,
        angles=angles,
        dso=dso,
        dsd=dsd,
        grid=grid,
        detector=detector,
        output_shape=input_shape,
        type_geometry=type_geometry,
        type_projector=type_projector,
        default_shape=op.inputs[0].shape[2:].as_list()
    )

    return [fp, None, None, None, None, None, None]


class DistanceDriven2DBP(tf.keras.layers.Layer):
    '''
    The keras module for backprojection.

    Initialization Parameters
    --------------------
    projector: ct_projector class.
        It holds the default geometries.
    angles: Tensor of shape [batch, nview], float32.
        Default projection angles in radius. Can be overloaded during call().
    type_geometry: TypeGeometry.
        Default geometry of the projection. Can be overloaded during call().
    type_projector: TypeProjector.
        Default projector. Can be overloaded during call().
        IR: where the ray intersection length is considered.
        FORCE_FBP: the ray intersection length is ignored. It is useful when calculating the conjugate to
            the backprojection in FBP.
    default_shape: list of int with length 3.
        It is corresponding to the image shape in [nz, ny, nx].
        Although the output shape can be passed during computation via output_shape, default_shape provides
        shape inference capability. Any element of default_shape can be -1, which means that the corresponding
        dimension will be fetched from output_shape during computation. The default_shape can also be None,
        then the nview will be derived from the geometry input, and nv/nu will be taken from projector.

    Call Parameters
    ------------------
    Mandatory:
    inputs: Tensor of shape [batch, nview, nv, nu, channel], float32.
        It is the projection to be backprojected.

    Optional:
    angles: Tensor of shape [batch, nview], float32.
        The projection angles in radius.
    dsd: Tensor of shape [batch, 1], float32.
        The distance from source to detector. If None, take the values from projector
    dso: Tensor of shape [batch, 1], float32.
        The distance from source to isocenter. If None, take the values from projector
    grid: Tensor of shape [batch, 6], float32.
        (dx, dy, dz, cx, cy, cz). If None, take the values from projector.
    detector: Tensor of shape [batch, 4], float32.
        (du, dv, off_u, off_v). If None, take the values from projector.
    output_shape: Tensor of shape [None, 3], int32.
        (nz, ny, nx). Only the first row will be taken for the output shape.
        If None, take the values from default_shape.
    default_shape: List of int with length 3.
        (nz, ny, nx). If None, take the values from projector.

    Returns
    ------------------
    img: Tensor of shape [batch, nz, ny, nx, channel], float32.
        The forward projection.

    '''
    def __init__(
        self,
        projector: ct_projector,
        angles: tf.Tensor = None,
        type_geometry: TypeGeometry = TypeGeometry.PARALLEL,
        type_projector: int = TypeProjector.IR,
        default_shape: List[int] = None,
        name: str = ''
    ):
        super(DistanceDriven2DBP, self).__init__(name=name)
        self.projector = projector
        self.angles = angles
        self.type_geometry = type_geometry
        self.type_projector = type_projector
        self.default_shape = default_shape

    def build(self, input_shape):
        super(DistanceDriven2DBP, self).build(input_shape)

    def call(self, inputs, **kwargs):
        kwargs_list = inspect.getfullargspec(distance_driven_2d_bp)[0]
        kwargs_call = {
            'angles': self.angles,
            'type_geometry': self.type_geometry,
            'type_projector': self.type_projector
        }
        for name in kwargs_list:
            if name in ['projector', 'prj', 'default_shape', 'name']:
                continue
            if name in kwargs:
                kwargs_call[name] = kwargs[name]

        return distance_driven_2d_bp(
            projector=self.projector,
            prj=inputs,
            default_shape=self.default_shape,
            **kwargs_call
        )
