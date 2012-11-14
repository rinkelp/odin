
"""
Tests: src/python/xray.py
"""

from odin import xray
from odin.testing import skip

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

class TestBeam():
    
    def setup():
        
        self.flux = 100.0
        
    
    def unit_convs():
        
        beam = xray.Beam(self.flux, energy=1.0)
        
        assert_almost_equal(beam.wavelength == 1.2398e4)
        assert_almost_equal(beam.frequency  == 2.4190e14)
        assert_almost_equal(beam.wavenumber == (2.0 * np.pi)/1.2398e4)
    
        
class TestDetector():
    
    def setup():
        self.d = xray.Detector.generic()
        self.spacing=0.05
        self.lim=10.0
        self.energy=0.7293
        self.flux=100.0
        self.l=50.0
    
    @skip
    def file_loading():
        
        self.ref_detector = xray.Detector.load('string')
    
    def recpolar_n_reciprocal():
        q1 = np.linalg.norm(self.d.reciprocal)
        q1 = self.recpolar[:,0]
        assert_array_almost_equal(q1, q2)
    
    def recpolar_space():
        """ try converting backwards """
        
        # this is the "generic" detector in real space
        x = np.arange(-self.lim, self.lim, self.spacing)
        xx, yy = np.meshgrid(x, x)

        xyz = np.zeros((len(x)**2, 3))
        xyz[:,0] = xx.flatten()
        xyz[:,1] = yy.flatten()
        
        # one slice along the horizontal direction in real space
        q = self.d.recpolar[:0]
        h = self.l * np.tan( np.arcsin(q/self.d.k) )
        assert_array_almost_equal(x, h)
        
        