'''
Wrapper for Tensorflow.
'''

import numpy as np
import configparser
import tensorflow as tf


# %%
def tile_tensor(tensor, batchsize):
    '''
    Tile the tensor if it only has one dimension
    '''
    if len(tensor.shape) == 1:
        return tf.tile(tensor[tf.newaxis], [batchsize, 1])
    else:
        return tensor


class ct_projector:
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

    def get_angles(self) -> np.array:
        '''
        Get the angles for each view in circular geometry.
        '''
        return np.arange(0, self.nview, dtype=np.float32) * 2 * np.pi / self.nview
