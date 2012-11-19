
"""
Tests: src/python/xray.py
"""

import os
from nose import SkipTest

from odin import xray, utils
from odin.testing import skip, ref_file, gputest
from mdtraj import trajectory, io

try:
    import gpuscatter
    GPU = True
except ImportError as e:
    GPU = False

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_allclose

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()


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
        self.spacing = 0.05
        self.lim     = 10.0
        self.energy  = 0.7293
        self.flux    = 100.0
        self.l       = 50.0
        self.d = xray.Detector.generic(spacing = self.spacing,
                                       lim = self.lim,
                                       energy = self.energy,
                                       flux = self.flux,
                                       l = self.l) 
    
    def test_recpolar_n_reciprocal(self):
        q1 = np.sqrt( np.sum( np.power(self.d.reciprocal,2), axis=1) )
        q2 = self.d.recpolar[:,0]
        assert_array_almost_equal(q1, q2)
        
    def test_polar_space(self):
        
        # this is the "generic" detector in real space
        x = np.arange(-self.lim, self.lim+self.spacing, self.spacing)
        xx, yy = np.meshgrid(x, x)
        
        # one slice along the horizontal direction in real space
        r     = self.d.polar[:,0]
        theta = self.d.polar[:,1]
        phi   = self.d.polar[:,2]
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        assert_array_almost_equal(xx.flatten(), x)
        assert_array_almost_equal(yy.flatten(), y)
   
    def test_reciprocal_space(self):
        qx = self.d.reciprocal[:,0]
        qy = self.d.reciprocal[:,1]
        
        Shat    = self.d._unit_vector(self.d.real)
        Sx_unit = Shat[:,0]
        Sy_unit = Shat[:,1]
        
        assert_array_almost_equal(qx/self.d.k, Sx_unit)
        assert_array_almost_equal(qy/self.d.k, Sy_unit)
   
    def test_recpolar_space(self):
                
        # build a reference conversion, using a different geometrical calc
        ref1 = np.zeros(self.d.xyz.shape)
        hd = np.sqrt( np.power(self.d.xyz[:,0], 2) + np.power(self.d.xyz[:,1], 2) )
        
        # consistency check on the polar conversion
        #theta = utils.arctan3( self.d.xyz[:,2], hd )
        #assert_array_almost_equal(theta, self.d.polar[:,1], err_msg='theta check wrong')
        
        # |q| = k*sqrt{ 2 - 2 cos(theta) }
        ref1[:,0] = self.d.k * np.sqrt( 2.0 - 2.0 * np.cos(self.d.polar[:,1]) )
                                
        # phi is the same as polar
        ref1[:,2] = self.d.polar[:,2].copy()
        
        assert_array_almost_equal( ref1[:,0], self.d.recpolar[:,0], err_msg='|q|')
        assert_array_almost_equal( ref1[:,2], self.d.recpolar[:,2], err_msg='phi')
        
        # NO THETA TEST -- DO WE CARE?
        
    def test_io(self):
        if os.path.exists('r.dtc'): os.system('rm r.dtc')
        self.d.save('r.dtc')
        d = xray.Detector.load('r.dtc')
        assert_array_almost_equal(d.xyz, self.d.xyz)
        if os.path.exists('r.dtc'): os.system('rm r.dtc') 
        
        
class TestShot():
    
    def setup(self):
        self.d = xray.Detector.generic(spacing=0.4)
        self.t = trajectory.load(ref_file('ala2.pdb'))
        self.shot = xray.Shot.load(ref_file('refshot.shot'))
        
    def test_io(self):
        if os.path.exists('test.shot'): os.system('test.shot')
        self.shot.save('test.shot')
        s = xray.Shot.load('test.shot')
        assert_array_almost_equal(s.intensity_profile(), self.shot.intensity_profile())
        os.remove('test.shot')
        if os.path.exists('test.shot'): os.system('test.shot')
    
    def test_sim(self):
        if not GPU: raise SkipTest
        shot = xray.Shot.simulate(self.t, 512, self.d)
        
    def test_polar_interpolation(self):
        pass
        
    @skip
    def test_mask(self):
        pass
        
    def test_nearests(self):
        pass
        
    def test_I_index(self):
        pass
        
    def test_correlation(self):
        
        # arb. parameters for testing
        q1 = 0.1
        q2 = 0.1
        delta = 45.0

        delta = self.shot._nearest_delta(delta)
        
        correlation = 0.0
        mean1 = 0.0
        mean2 = 0.0
        
        for phi in self.shot.phi_values:
            
            if (  (q1,phi) not in self.shot._masked_coords ) and (  (q2,phi+delta) not in self.shot._masked_coords ):            
                x = self.shot.I(q1, phi)
                y = self.shot.I(q2, phi+delta)
                mean1 += x
                mean2 += y
                correlation += x*y
        
        correlation /= float(self.shot.num_phi)
        mean1 /= float(self.shot.num_phi)
        mean2 /= float(self.shot.num_phi)
        
        ref = correlation - (mean1*mean2)
        
        assert_almost_equal(ref, self.shot.correlate(q1, q2, delta))
    
    
    def test_corr_ring(self):
        
        q1 = 1.0
        q2 = 1.0
        
        # recall the possible deltas are really the possible values of phi
        correlation_ring = np.zeros(( self.shot.num_phi, 2 ))
        correlation_ring[:,0] = np.array(self.shot.phi_values)
        
        # now just correlate for each value of delta
        for i in range(self.shot.num_phi):
            
            delta = self.shot.polar_intensities[i,1] # phi value
            delta = self.shot._nearest_delta(delta)

            correlation = 0.0
            mean1 = 0.0
            mean2 = 0.0

            for phi in self.shot.phi_values:

                if (  (q1,phi) not in self.shot._masked_coords ) and (  (q2,phi+delta) not in self.shot._masked_coords ):            
                    x = self.shot.I(q1, phi)
                    y = self.shot.I(q2, phi+delta)
                    mean1 += x
                    mean2 += y
                    correlation += x*y

            correlation /= float(self.shot.num_phi)
            mean1 /= float(self.shot.num_phi)
            mean2 /= float(self.shot.num_phi)

            ref = correlation - (mean1*mean2)
            
            correlation_ring[i,1] = ref
        
        assert_array_almost_equal(correlation_ring, self.shot.correlate_ring(q1, q2))

    def test_simulate(self):
        if not GPU: raise SkipTest
        d = xray.Detector.generic(spacing=0.4)
        x = xray.Shot.simulate(self.t, 512, d)
        
        
class TestShotset():
    
    def setup(self):
        self.shot = xray.Shot.load(ref_file('refshot.shot'))
        self.shotset = xray.Shotset([self.shot])
        self.t = trajectory.load(ref_file('ala2.pdb'))
        
    def test_simulate(self):
        if not GPU: raise SkipTest
        d = xray.Detector.generic(spacing=0.4)
        x = xray.Shotset.simulate(self.t, 512, d, 2)
    
    def test_detector_checking(self):
        if not GPU: raise SkipTest
        d1 = xray.Detector.generic(spacing=0.4)
        d2 = xray.Detector.generic(spacing=0.4)

        s1 = xray.Shot.simulate(self.t, 512, d1)
        s2 = xray.Shot.simulate(self.t, 512, d2)
        s2.interpolate_to_polar(phi_spacing=2.0)
        
        try:
            st = xray.Shotset([s1, s2])
            raise RuntimeError()
        except ValueError as e:
            print e
            return # this means success
        
    def test_intensities(self):
        q = 0.1
        i1 = self.shot.qintensity(q)
        i2 = self.shotset.qintensity(q)
        assert_almost_equal(i1, i2)
        
    def test_profiles(self):
        i1 = self.shot.intensity_profile()
        i2 = self.shotset.intensity_profile()
        assert_array_almost_equal(i1, i2)
        
    def test_correlations(self):
        q1 = 0.1
        q2 = 0.1
        delta = 45.0
        i1 = self.shot.correlate(q1, q2, delta)
        i2 = self.shotset.correlate(q1, q2, delta)
        assert_almost_equal(i1, i2)
        
    def test_rings(self):
        q1 = 0.1
        q2 = 0.1
        i1 = self.shot.correlate_ring(q1, q2)
        i2 = self.shotset.correlate_ring(q1, q2)
        assert_array_almost_equal(i1, i2)

if __name__ == '__main__':
    test = TestDetector()
    test.setup()
    #test.test_recpolar_n_reciprocal()
    test.test_recpolar_space()
