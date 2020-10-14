from ctypes import *
import os
import cupy as cp
import numpy as np
import re

module = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'libprojector.so'))

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
        with open(filename, 'r') as f:    
            for line in f:
                tokens = line.strip('\n').split('=')
                if len(tokens) == 2:
                    setattr(self, tokens[0].lower(), np.float32(tokens[1]))
        self.nview = int(self.nview)
        self.nu = int(self.nu)
        self.nv = int(self.nv)
        self.nx = int(self.nx)
        self.ny = int(self.ny)
        self.nz = int(self.nz)
    
    def set_device(self, device):
        return module.SetDevice(c_int(device))
    
    def siddon_cone_fp_abitrary(self, img, det_center, det_u, det_v, src):
        '''
        Conebeam forward projection with abitrary geomety and flat panel. Using Siddon ray tracing
        @params:
        @img: the image to be projected, of size [batch, nz, ny, nx]
        @det_center: the center of the detector in mm, of size [nview, 3]
        @det_u: the u axis of the detector, normalized, of size [nview, 3]
        @det_v: the v axis of the detector, normalized, of size [nview, 3]
        @src: the src positions, in mm, of size [nview, 3]
        
        @return: 
        @prj: the forward projection, of size [batch, nview, self.nv, self.nu]
        
        batch, nx, ny, nz, nview will be automatically derived from the parameters. The projection size should be set by self.nu and self.nv
        '''
        
        # projection of size 
        prj = cp.zeros([img.shape[0], det_center.shape[0], self.nv, self.nu], cp.float32)
        
        module.cupySiddonConeProjectionAbitrary.restype = c_int

        err = module.cupySiddonConeProjectionAbitrary(
            c_void_p(prj.data.ptr), 
            c_void_p(img.data.ptr), 
            c_void_p(det_center.data.ptr), 
            c_void_p(det_u.data.ptr), 
            c_void_p(det_v.data.ptr), 
            c_void_p(src.data.ptr),
            c_int(img.shape[0]), 
            c_int(img.shape[3]), c_int(img.shape[2]), c_int(img.shape[1]), 
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_float(self.cx), c_float(self.cy), c_float(self.cz),
            c_int(prj.shape[3]), c_int(prj.shape[2]), c_int(prj.shape[1]),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        if err != 0:
            print (err)
        
        return prj
    
    def siddon_cone_bp_abitrary(self, prj, det_center, det_u, det_v, src):
        '''
        Conebeam backprojection with abitrary geomety and flat panel. Using Siddon ray tracing
        @params:
        @prj: the projection to be backprojected, of size [batch, nview, nv, nu]
        @det_center: the center of the detector in mm, of size [nview, 3]
        @det_u: the u axis of the detector, normalized, of size [nview, 3]
        @det_v: the v axis of the detector, normalized, of size [nview, 3]
        @src: the src positions, in mm, of size [nview, 3]
        
        @return: 
        @img: the backprojection, of size [batch, self.nz, self.ny, self.nx]
        
        batch, nu, nv, nview will be automatically derived from the parameters. The image size should be set by self.nx, self.ny, and self.nz
        '''
        
        # make sure they are float32
        
        # projection of size 
        img = cp.zeros([prj.shape[0], self.nz, self.ny, self.nx], cp.float32)
        
        module.cupySiddonConeBackprojectionAbitrary.restype = c_int

        err = module.cupySiddonConeBackprojectionAbitrary(
            c_void_p(img.data.ptr), 
            c_void_p(prj.data.ptr), 
            c_void_p(det_center.data.ptr), 
            c_void_p(det_u.data.ptr), 
            c_void_p(det_v.data.ptr), 
            c_void_p(src.data.ptr),
            c_int(img.shape[0]), 
            c_int(img.shape[3]), c_int(img.shape[2]), c_int(img.shape[1]), 
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_float(self.cx), c_float(self.cy), c_float(self.cz),
            c_int(prj.shape[3]), c_int(prj.shape[2]), c_int(prj.shape[1]),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        if err != 0:
            print (err)

        return img
