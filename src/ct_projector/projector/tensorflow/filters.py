'''
Reconstruction filters, implemented by tf.signal.fft
'''

# %%
import numpy as np
import tensorflow as tf
from enum import IntEnum

from .circular_2d import TypeGeometry


# %%
class TypeFilter(IntEnum):
    RL = 0
    HANN = 2


# %%
def is_equal_space(type_geometry: TypeGeometry):
    '''
    If the filter is equal-space or equal-angular.
    '''
    if type_geometry in [TypeGeometry.PARALLEL]:
        return True
    else:
        return False


def get_ramp_filter(
    nu: int,
    du: float,
    type_filter: TypeFilter = TypeFilter.RL,
    type_geometry: TypeGeometry = TypeGeometry.PARALLEL
):
    '''
    Return a ramp filter
    '''
    # Get the ramp filter in the spatial domain
    filter_len = 2 * nu - 1
    ramp = np.zeros([filter_len], np.float32)
    k = np.arange(filter_len) - (nu - 1)
    # central point
    ramp[k == 0] = 1 / (4 * du * du)
    # odd points
    inds = np.where(k % 2 == 1)
    if is_equal_space(type_geometry):
        w = np.pi * k[inds] * du
    else:
        w = np.pi * np.sin(k[inds] * du)
    ramp[inds] = -1 / (w * w)

    # apply additional window
    if type_filter == TypeFilter.HANN:
        window_freq = 0.5 + 0.5 * np.cos(2 * np.pi * np.arange(filter_len) / filter_len)
    else:
        window_freq = np.ones([filter_len])
    ramp_freq = np.fft.fft(ramp) * window_freq
    ramp = np.fft.ifft(ramp_freq).real

    return ramp


# %%
class ProjectionFilter(tf.keras.layers.Layer):
    '''
    The keras module for projection filter
    '''
    def __init__(
        self,
        du: float,
        type_geometry: TypeGeometry = TypeGeometry.PARALLEL,
        type_filter: TypeFilter = TypeFilter.RL,
        name: str = ''
    ):
        super(ProjectionFilter, self).__init__(name=name)
        self.du = du
        self.type_geometry = type_geometry
        self.type_filter = type_filter

    def build(self, input_shape):
        '''
        Allocate filter

        input_shape is [batch, nview, nv, nu, channel]
        '''
        super(ProjectionFilter, self).build(input_shape)
        self.nview = input_shape[1]
        self.nu = input_shape[-2]
        kernel = get_ramp_filter(self.nu, self.du, self.type_filter, self.type_geometry)
        freq_kernel = np.fft.rfft(kernel)
        self.freq_kernel_tensor = tf.convert_to_tensor(freq_kernel, tf.complex64)
        self.fft_length = tf.convert_to_tensor(np.array([self.nu * 2 - 1]), tf.int32)

    def call(self, inputs, **kwargs):
        '''
        inputs is [batch, nview, nv, nu, channel]
        '''
        # convert to [batch, channel, nview, nv, nu]
        inputs = tf.transpose(inputs, [0, 4, 1, 2, 3])

        # filter
        freq_prj = tf.signal.rfft(inputs, self.fft_length)
        freq_prj *= self.freq_kernel_tensor
        fprj = tf.signal.irfft(freq_prj)
        # truncation
        fprj = fprj[..., self.nu - 1:]

        # scale
        scale = np.pi / self.nview * self.du
        fprj *= scale

        # convert back to [batch, nview, nv, nu, channel]
        fprj = tf.transpose(fprj, [0, 2, 3, 4, 1])

        return fprj
