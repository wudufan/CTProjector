import os
import numpy as np
import configparser
import tensorflow as tf

module = tf.load_op_library(os.path.join(os.path.dirname(__file__), 'libtfprojector.so'))

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

    def siddon_cone_fp_abitrary(self, img, geometry, grid = None, detector = None, output_shape = None, default_shape = None, name = None):
        '''
        Conebeam forward projection with abitrary geomety and flat panel. Using Siddon ray tracing
        @params:
        @img: tensor of shape [batch, nx, ny, nz, channel], the image to be projected. Type float32
        @geometry: tensor of shape [batch, nview*4, 3], corresponding to det_center, det_u, det_v and src. Type float32
        @grid: tensor of shape [batch, 6]: (dx, dy, dz, cx, cy, cz). If None, take the values from class attributes. Type float32
        @detector: tensor of shape [batch, 4]: (du, dv, off_u, off_v). If None, take the values from class attibutes. Type float32
        @output_shape: tensor of shape [None, 3]: (nview, nv, nu). Only the first row will be taken for the output shape. If None, take the values from default_shape. Type int32
        @default_shape: list of length 3: (nview, nv, nu). If None, take the values from class attributes. Type int32
        @name: name of the op

        @return: 
        @prj: tensor of shape [batch, nu, nv, nview, channel]. The forward projection. Type float32
        '''
        batchsize = tf.shape(img)[0]

        if grid is None:
            grid = np.array([self.dx, self.dy, self.dz, self.cx, self.cy, self.cz], np.float32)
        if detector is None:
            detector = np.array([self.du, self.dv, self.off_u, self.off_v], np.float32)
        if default_shape is None:
            default_shape = [geometry.shape[1]//4, self.nv, self.nu]
        if output_shape is None:
            output_shape = np.copy(default_shape).astype(np.int32)
        
        grid = tile_tensor(grid, batchsize)
        detector = tile_tensor(detector, batchsize)
        output_shape = tile_tensor(output_shape, batchsize)
        
        # (batch, channel, nz, ny, nx)
        img = tf.transpose(img, (0, 4, 3, 2, 1))

        prj = module.siddon_cone_fp(image = img, geometry = geometry, grid = grid, detector = detector, output_shape = output_shape, default_shape = default_shape, name = name)
        
        # reshape it back
        # (batch, nu, nv, nview, channel)
        prj = tf.transpose(prj, (0, 4, 3, 2, 1))
        
        return prj
    
    def siddon_cone_bp_abitrary(self, prj, geometry, grid = None, detector = None, output_shape = None, default_shape = None, name = None):
        '''
        Conebeam backprojection with abitrary geomety and flat panel. Using Siddon ray tracing
        @params:
        @prj: tensor of shape [batch, nu, nv, nview, channel], the projection to be backprojected. Type float32
        @geometry: tensor of shape [batch, nview*4, 3], corresponding to det_center, det_u, det_v and src. Type float32
        @grid: tensor of shape [batch, 6]: (dx, dy, dz, cx, cy, cz). If None, take the values from class attributes. Type float32
        @detector: tensor of shape [batch, 4]: (du, dv, off_u, off_v). If None, take the values from class attibutes. Type float32
        @output_shape: tensor of shape [None, 3]: (nview, nv, nu). Only the first row will be used. If None, take the values from default_shape. Type int32
        @default_shape: list of length 3: (nview, nv, nu). If None, take the values from class attributes. Type int32
        @name: name of the op

        @return: 
        @img: tensor of shape [batch, nx, ny, nz, channel]. The backprojected image. Type float32
        '''
        batchsize = tf.shape(prj)[0]

        if grid is None:
            grid = np.array([self.dx, self.dy, self.dz, self.cx, self.cy, self.cz], np.float32)
        if detector is None:
            detector = np.array([self.du, self.dv, self.off_u, self.off_v], np.float32)
        if default_shape is None:
            default_shape = [self.nx, self.ny, self.nz]
        if output_shape is None:
            output_shape = np.copy(default_shape).astype(np.int32)
        
        grid = tile_tensor(grid, batchsize)
        detector = tile_tensor(detector, batchsize)
        output_shape = tile_tensor(output_shape, batchsize)
        
        # reshape the img tensor for input
        # (batch, channel, nview, nv, nu)
        prj = tf.transpose(prj, (0, 4, 3, 2, 1))

        img = module.siddon_cone_bp(projection = prj, geometry = geometry, grid = grid, detector = detector, output_shape = output_shape, default_shape = default_shape, name = name)

        # reshape it back
        # (batch, nx, ny, nz, channel)
        img = tf.transpose(img, (0, 4, 3, 2, 1))
        
        return img
    

# register gradients
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

@ops.RegisterGradient("SiddonConeFP")
def _siddon_cone_fp_grad(op, grad):
    '''
    The gradients for SiddonConeFP 

    @params:
    @op: the SiddonConeFP operator, where we can find original information
    @grad: the gradient w.r.t. output of SiddonConeFP

    @return:
    Gradients w.r.t. input of SiddonConeFP. Only the gradient to the image will be returned, others will be set to None
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
    bp = module.siddon_cone_bp(projection = grad, geometry = geometry, grid = grid, detector = detector, output_shape = input_shape, default_shape = op.inputs[0].shape[2:].as_list())

    return [bp, None, None, None, None]

@ops.RegisterGradient("SiddonConeBP")
def _siddon_cone_bp_grad(op, grad):
    '''
    The gradients for SiddonConeBP 

    @params:
    @op: the SiddonConeBP operator, where we can find original information
    @grad: the gradient w.r.t. output of SiddonConeBP

    @return:
    Gradients w.r.t. input of SiddonConeBP. Only the gradient to the projection will be returned, others will be set to None
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
    fp = module.siddon_cone_fp(image = grad, geometry = geometry, grid = grid, detector = detector, output_shape = input_shape, default_shape = op.inputs[0].shape[2:].as_list())

    return [fp, None, None, None, None]
    

# keras modules
class SiddonConeFP(tf.keras.layers.Layer):
    '''
    The keras module for forward projection.

    @Intializing params:
    @projector: a ct_projector class which holds the default geometries
    @default_shape: a list of int with length of 3. It is corresponding to the forward projection shape in [nview, nv, nu]. 
        Although the output shape can be passed during computation via output_shape, default_shape provides shape inference capability. 
        Any element of default_shape can be -1, which means that the corresponding dimension will be fetched from output_shape during computation.
        The default_shape can also be None, then the nview will be derived from the geometry input, and nv/nu will be taken from projector. 
    
    @Call params:
    @Mandatory:
    @inputs: a float32 tensor of shape [batch, nx, ny, nz, channel]. It is the image to be forward projected
    @geometry: a float32 tensor of shape [batch, nview * 4, 3]. Different geometries can be set to different elements in one batch. 
        geometry should be passed by kwargs only, i.e. SiddonConeFP(...)(..., geometry = geometry)
        For each elements (geometry[i]), it consists of the following 4 part, each with size [nview, 3]:
        @det_centers: the central position of the detector. (x, y, z) at each view;
        @det_us: the axis of the U direction of the detector in the world coordinates. (x, y, z) at each view. It should be normalized. 
        @det_vs: the axis of the V direction of the detector in the world coordinates. (x, y, z) at each view. It should be normalized and perpendicular to det_us.
        @srcs: the source position, (x, y, z) at each view.
    @optional:
    @grid: a float32 tensor of shape [batch, 6] or [6]. Each elements corresponding to (dx, dy, dz, cx, cy, cz) in mm. 
        If it is in [batch, 6], each row sets a different grid for the corresponding input image. 
        If it is [6], it sets the same grid for all the input images. 
        If it is not presented or None, will take value from self.projector. 
    @detector: a float32 tensor of shape [batch, 4] or [4]. Each elements corresponding to (du, dv, off_u, off_v). du/dv are in mm and off_u/off_v are in pixels.
        If it is in [batch, 4], each row sets a different detector for the corresponding output projection. 
        If it is [4], it sets the same detector for all the output pojections. 
        If it is not presented or None, will take value from self.projector. 
    @output_shape: a float32 tensor of shape [None, 3] or [3]. Each elements corresponding to (nview, nv, nu). 
        Only the first row in output_shape will be used as the output shape. 
        If it is not presented or None, will take value from default_shape. 
    
    @Return:
    The forward projection. It is a float32 tensor of shape [batch, nu, nv, nview, channel]

    '''
    def __init__(self, projector, default_shape = None, name = ''):
        super(SiddonConeFP, self).__init__(name=name)
        self.projector = projector
        self.default_shape = default_shape
    
    def build(self, input_shape):
        super(SiddonConeFP, self).build(input_shape)

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
        
        return self.projector.siddon_cone_fp_abitrary(inputs, geometry, grid, detector, output_shape, self.default_shape)

class SiddonConeBP(tf.keras.layers.Layer):
    def __init__(self, projector, default_shape = None, name = ''):
        super(SiddonConeBP, self).__init__(name=name)
        self.projector = projector
        self.default_shape = default_shape
    
    def build(self, input_shape):
        super(SiddonConeBP, self).build(input_shape)

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
        
        return self.projector.siddon_cone_bp_abitrary(inputs, geometry, grid, detector, output_shape, self.default_shape)