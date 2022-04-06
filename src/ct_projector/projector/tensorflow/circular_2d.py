'''
Circular 2D geometry
'''

# %%
import numpy as np
import tensorflow as tf
from enum import IntEnum
from typing import List

from .ct_projector import ct_projector, tile_tensor

import pkg_resources

module = tf.load_op_library(
    pkg_resources.resource_filename('ct_projector', 'kernel/bin/libtfprojector.so')
)


# %%
class TypeGeometry(IntEnum):
    PARALLEL = 1
    FAN_EQUIANGULAR = 2


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
    type_geometry: TypeGeometry = TypeGeometry.FAN_EQUIANGULAR,
    name: str = None
) -> tf.Tensor:
    '''
    Conebeam forward projection with arbitrary geometry and flat panel. Using Siddon ray tracing.

    Parameters
    ----------------
    img: Tensor of shape [batch, nx, ny, nz, channel], float32.
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
    name: str.
        Name of the operation

    Returns
    ----------------
    prj: Tensor of shape [batch, nu, nv, nview, channel], float32.
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
        detector = np.array([projector.du / projector.dsd, projector.dv, projector.off_u, projector.off_v], np.float32)
    if default_shape is None:
        default_shape = [angles.shape[1], projector.nv, projector.nu]
    if output_shape is None:
        output_shape = np.copy(default_shape).astype(np.int32)

    grid = tile_tensor(grid, batchsize)
    detector = tile_tensor(detector, batchsize)
    output_shape = tile_tensor(output_shape, batchsize)

    # (batch, channel, nz, ny, nx)
    img = tf.transpose(img, (0, 4, 3, 2, 1))

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
        name=name
    )

    # reshape it back
    # (batch, nu, nv, nview, channel)
    prj = tf.transpose(prj, (0, 4, 3, 2, 1))

    return prj
