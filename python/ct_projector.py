from ctypes import *
import os
import numpy as np
import re
import scipy.ndimage
import configparser

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
        
        # make sure they are float32
        img = img.astype(np.float32)
        det_center = det_center.astype(np.float32)
        det_u = det_u.astype(np.float32)
        det_v = det_v.astype(np.float32)
        src = src.astype(np.float32)
        
        # projection of size 
        prj = np.zeros([img.shape[0], det_center.shape[0], self.nv, self.nu], np.float32)
        
        module.cSiddonConeProjectionAbitrary.restype = c_int

        err = module.cSiddonConeProjectionAbitrary(
            prj.ctypes.data_as(POINTER(c_float)), 
            img.ctypes.data_as(POINTER(c_float)), 
            det_center.ctypes.data_as(POINTER(c_float)), 
            det_u.ctypes.data_as(POINTER(c_float)), 
            det_v.ctypes.data_as(POINTER(c_float)), 
            src.ctypes.data_as(POINTER(c_float)),
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
        prj = prj.astype(np.float32)
        det_center = det_center.astype(np.float32)
        det_u = det_u.astype(np.float32)
        det_v = det_v.astype(np.float32)
        src = src.astype(np.float32)
        
        # projection of size 
        img = np.zeros([prj.shape[0], self.nz, self.ny, self.nx], np.float32)
        
        module.cSiddonConeBackprojectionAbitrary.restype = c_int

        err = module.cSiddonConeBackprojectionAbitrary(
            img.ctypes.data_as(POINTER(c_float)), 
            prj.ctypes.data_as(POINTER(c_float)), 
            det_center.ctypes.data_as(POINTER(c_float)), 
            det_u.ctypes.data_as(POINTER(c_float)), 
            det_v.ctypes.data_as(POINTER(c_float)), 
            src.ctypes.data_as(POINTER(c_float)),
            c_int(img.shape[0]), 
            c_int(img.shape[3]), c_int(img.shape[2]), c_int(img.shape[1]), 
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_float(self.cx), c_float(self.cy), c_float(self.cz),
            c_int(prj.shape[3]), c_int(prj.shape[2]), c_int(prj.shape[1]),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        if err != 0:
            print (err)

        return img
    
    def distance_driven_fp_tomo(self, img, det_center, src):
        '''
        Distance driven forward projection for tomosynthesis. It assumes that the detector has u=(1,0,0) and v = (0,1,0).
        The projection should be along the z-axis (main axis for distance driven projection)
        
        @params:
        @img: original image, of size (batch, nz, ny, nx)
        @det_center: centers of the detector for each projection, of size (nview, 3)
        @src: position of the source for each projection, of size (nview ,3)

        @return:
        @prj: the forward projection of size (batch, nview, nv, nu)
        '''
        img = img.astype(np.float32)
        det_center = det_center.astype(np.float32)
        src = src.astype(np.float32)
        
        prj = np.zeros([img.shape[0], det_center.shape[0], self.nv, self.nu], np.float32)

        module.cDistanceDrivenTomoProjection.restype = c_int
        err = module.cDistanceDrivenTomoProjection(
            prj.ctypes.data_as(POINTER(c_float)), 
            img.ctypes.data_as(POINTER(c_float)), 
            det_center.ctypes.data_as(POINTER(c_float)), 
            src.ctypes.data_as(POINTER(c_float)),
            c_int(img.shape[0]),
            c_int(img.shape[3]), c_int(img.shape[2]), c_int(img.shape[1]), 
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_float(self.cx), c_float(self.cy), c_float(self.cz),
            c_int(prj.shape[3]), c_int(prj.shape[2]), c_int(prj.shape[1]),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        if err != 0:
            print (err)

        return prj
    
    def distance_driven_bp_tomo(self, prj, det_center, src):
        '''
        Distance driven backprojection for tomosynthesis. It assumes that the detector has u=(1,0,0) and v = (0,1,0).
        The backprojection should be along the z-axis (main axis for distance driven projection)

        @params:
        @prj: the projection to BP, of size (batch, nview, nv, nu)
        @det_center: centers of the detector for each projection, of size (nview, 3)
        @src: position of the source for each projection, of size (nview ,3)

        @return:
        @img: the backprojection image, of size (batch, nz, ny, nx)
        '''

        prj = prj.astype(np.float32)
        det_center = det_center.astype(np.float32)
        src = src.astype(np.float32)
        
        img = np.zeros([prj.shape[0], self.nz, self.ny, self.nx], np.float32)

        module.cDistanceDrivenTomoBackprojection.restype = c_int
        err = module.cDistanceDrivenTomoBackprojection(
            img.ctypes.data_as(POINTER(c_float)), 
            prj.ctypes.data_as(POINTER(c_float)), 
            det_center.ctypes.data_as(POINTER(c_float)), 
            src.ctypes.data_as(POINTER(c_float)),
            c_int(img.shape[0]),
            c_int(img.shape[3]), c_int(img.shape[2]), c_int(img.shape[1]), 
            c_float(self.dx), c_float(self.dy), c_float(self.dz),
            c_float(self.cx), c_float(self.cy), c_float(self.cz),
            c_int(prj.shape[3]), c_int(prj.shape[2]), c_int(prj.shape[1]),
            c_float(self.du), c_float(self.dv), c_float(self.off_u), c_float(self.off_v))
        
        if err != 0:
            print (err)

        return img

