
"""
Tests: src/python/xray.py
"""

import os

from odin import xray
from odin.testing import skip, ref_file

try:
    import gpuscatter
    GPU = True
except ImportError as e:
    GPU = False

from mdtraj import trajectory, io

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_allclose

class TestBeam():
    
    def setup(self):
        
        self.flux = 100.0
        
    
    def test_unit_convs(self):
        
        beam = xray.Beam(self.flux, energy=1.0)
        
        assert_allclose(beam.wavelength, 12.398, rtol=1e-3)
        assert_allclose(beam.frequency, 2.4190e17, rtol=1e-3)
        assert_allclose(beam.wavenumber, (2.0 * np.pi)/12.398, rtol=1e-3)
    
        
class TestDetector():
    
    def setup(self):
        self.d = xray.Detector.generic()
        self.spacing=0.05
        self.lim=10.0
        self.energy=0.7293
        self.flux=100.0
        self.l=50.0
    
    @skip
    def test_file_loading(self):
        
        self.ref_detector = xray.Detector.load('string')
    
    def test_recpolar_n_reciprocal(self):
        q1 = np.sqrt( np.sum( np.power(self.d.reciprocal,2), axis=1) )
        q2 = self.d.recpolar[:,0]
        assert_array_almost_equal(q1, q2)
   
    @skip 
    def test_recpolar_space(self):
        """ try converting backwards """
        
        # this is the "generic" detector in real space
        x = np.arange(-self.lim, self.lim, self.spacing)
        xx, yy = np.meshgrid(x, x)

        xyz = np.zeros((len(x)**2, 3))
        xyz[:,0] = xx.flatten()
        xyz[:,1] = yy.flatten()
        
        # one slice along the horizontal direction in real space
        q = self.d.reciprocal[:,0]
        h = self.l * np.tan( 2.0 * np.arcsin(q/self.d.k) )
        assert_array_almost_equal(x, h)
        
    def test_io(self):
        if os.path.exists('r.dtc'): os.system('rm r.dtc')
        self.d.save('r.dtc')
        d = xray.Detector.load('r.dtc')
        assert_array_almost_equal(d.xyz, self.d.xyz)
        if os.path.exists('r.dtc'): os.system('rm r.dtc') 
        
class TestShot():
    
    def setup(self):
        self.d = xray.Detector.generic(spacing=0.3)
        self.t = trajectory.load(ref_file('ala2.pdb'))
    
    def test_sim(self):
        if not GPU: raise SkipTest
        self.shot = xray.Shot.simulate(self.t, 512, self.d)
        
        

