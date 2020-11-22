import os
import numpy as np
import configparser
import tensorflow as tf

module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'libtfprojector.so'))

class ct_projector:
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

    def from_file(self, filename):
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

    def make_geometry(self, det_center, det_u, det_v, src):
        '''
        Make the numpy input to the geometry required for siddon_cone_fp(bp)_abitrary
        '''

        return np.concatenate((det_center, det_u, det_v, src), 0).astype(np.float32)

    def siddon_cone_fp_abitrary(self, img, geometry, grid = None, detector = None, output_shape = None, default_shape = None):
        '''
        Conebeam forward projection with abitrary geomety and flat panel. Using Siddon ray tracing
        @params:
        @img: tensor of shape [batch, nx, ny, nz, channel], the image to be projected. Type float32
        @geometry: tensor of shape [nview*4, 3], corresponding to det_center, det_u, det_v and src. Type float32
        @grid: tensor of length 6: (dx, dy, dz, cx, cy, cz). If None, take the values from class attributes. Type float32
        @detector: tensor of length 4: (du, dv, off_u, off_v). If None, take the values from class attibutes. Type float32
        @output_shape: tensor of length 3: (nview, nv, nu). If None, take the values from default_shape. Type int32
        @default_shape: tensor of length 3: (nview, nv, nu). If None, take the values from class attributes. Type int32

        @return: 
        @prj: tensor of shape [batch, nu, nv, nview, channel]. The forward projection. Type float32
        '''

        if grid is None:
            grid = np.array([self.dx, self.dy, self.dz, self.cx, self.cy, self.cz], np.float32)
        if detector is None:
            detector = np.array([self.du, self.dv, self.off_u, self.off_v], np.float32)
        if default_shape is None:
            default_shape = [geometry.shape[0]//4, self.nv, self.nu]
        if output_shape is None:
            output_shape = np.copy(default_shape)
        
        # reshape the img tensor for input
        nchannels = img.shape[-1]
        # (batch, channel, nz, ny, nx)
        img = tf.transpose(img, (0, 4, 3, 2, 1))
        # (batch * channel, nz, ny, nx)
        img = tf.reshape(img, (-1, img.shape[2], img.shape[3], img.shape[4]))

        prj = module.siddon_cone_fp(image = img, geometry = geometry, grid = grid, detector = detector, output_shape = output_shape, default_shape = default_shape)

        # reshape it back
        # (batch, channel, nview, nv, nu)
        prj = tf.reshape(prj, (-1, nchannels, prj.shape[1], prj.shape[2], prj.shape[3]))
        # (batch, nu, nv, nview, channel)
        prj = tf.transpose(prj, (0, 4, 3, 2, 1))
        
        return prj
    
    def siddon_cone_bp_abitrary(self, prj, geometry, grid = None, detector = None, output_shape = None, default_shape = None):
        '''
        Conebeam backprojection with abitrary geomety and flat panel. Using Siddon ray tracing
        @params:
        @prj: tensor of shape [batch, nu, nv, nview, channel], the projection to be backprojected. Type float32
        @geometry: tensor of shape [nview*4, 3], corresponding to det_center, det_u, det_v and src. Type float32
        @grid: tensor of length 6: (dx, dy, dz, cx, cy, cz). If None, take the values from class attributes. Type float32
        @detector: tensor of length 4: (du, dv, off_u, off_v). If None, take the values from class attibutes. Type float32
        @output_shape: tensor of length 3: (nview, nv, nu). If None, take the values from default_shape. Type int32
        @default_shape: tensor of length 3: (nview, nv, nu). If None, take the values from class attributes. Type int32

        @return: 
        @img: tensor of shape [batch, nx, ny, nz, channel]. The backprojected image. Type float32
        '''

        if grid is None:
            grid = np.array([self.dx, self.dy, self.dz, self.cx, self.cy, self.cz], np.float32)
        if detector is None:
            detector = np.array([self.du, self.dv, self.off_u, self.off_v], np.float32)
        if default_shape is None:
            default_shape = [self.nz, self.ny, self.nx]
        if output_shape is None:
            output_shape = np.copy(default_shape)
        
        # reshape the img tensor for input
        nchannels = prj.shape[-1]
        # (batch, channel, nview, nv, nu)
        prj = tf.transpose(prj, (0, 4, 3, 2, 1))
        # (batch * channel, nview, nv, nu)
        prj = tf.reshape(prj, (-1, prj.shape[2], prj.shape[3], prj.shape[4]))

        img = module.siddon_cone_bp(projection = prj, geometry = geometry, grid = grid, detector = detector, output_shape = output_shape, default_shape = default_shape)

        # reshape it back
        # (batch, channel, nz, ny, nx)
        img = tf.reshape(img, (-1, nchannels, img.shape[1], img.shape[2], img.shape[3]))
        # (batch, nx, ny, nz, channel)
        img = tf.transpose(img, (0, 4, 3, 2, 1))
        
        return img