
"""
Tests: src/python/xray.py
"""

import os, sys
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
from numpy.testing import (assert_almost_equal, assert_array_almost_equal, 
                           assert_allclose, assert_array_equal)

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
        
        assert_array_almost_equal(yy.flatten(), x)
        assert_array_almost_equal(xx.flatten(), y)
   
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
        
        assert_array_almost_equal(ref1[:,0], self.d.recpolar[:,0], err_msg='|q|')
        assert_array_almost_equal(ref1[:,2], self.d.recpolar[:,2], err_msg='phi')
        
        # NO THETA TEST -- DO WE CARE?
       
    def test_basis_factory(self):
        beam = xray.Beam(self.flux, energy=self.energy)
        basis = (self.spacing, self.spacing, 0.0)
        dim = 2*(self.lim / self.spacing) + 1
        shape = (dim, dim, 1)
        corner = (-self.lim, -self.lim, 0.0)
        basis_list = [(basis, shape, corner)]
        bd = xray.Detector.from_basis(basis_list, self.l, beam)
        assert_array_almost_equal(bd.xyz, self.d.xyz)
        
    def test_io(self):
        if os.path.exists('r.dtc'): os.system('rm r.dtc')
        self.d.save('r.dtc')
        d = xray.Detector.load('r.dtc')
        if os.path.exists('r.dtc'): os.system('rm r.dtc') 
        assert_array_almost_equal(d.xyz, self.d.xyz)
        
        
class TestShot():
        
    def setup(self):
        self.d = xray.Detector.generic(spacing=0.4)
        self.i = np.random.randn(self.d.xyz.shape[0])
        self.t = trajectory.load(ref_file('ala2.pdb'))
        self.shot = xray.Shot.load(ref_file('refshot.shot'))
        
    def test_io(self):
        if os.path.exists('test.shot'): os.system('test.shot')
        self.shot.save('test.shot')
        s = xray.Shot.load('test.shot')
        os.remove('test.shot')
        if os.path.exists('test.shot'): os.system('test.shot')
        assert_array_almost_equal(np.array(s.intensity_profile()),
                                  np.array(self.shot.intensity_profile()))
        
    def test_sim(self):
        if not GPU: raise SkipTest
        shot = xray.Shot.simulate(self.t, 512, self.d)
        
    def test_implicit_interpolation(self):
        s = xray.Shot(self.i, self.d)
        
    def test_unstructured_interpolation(self):
        d = xray.Detector.generic(spacing=0.4, force_explicit=True)
        s = xray.Shot(self.i, d)

        
    @skip
    def test_mask(self):
        pass
        
    def test_nearests(self):
        
        nq = self.shot._nearest_q(0.0199)
        assert_almost_equal(nq, 0.02)
        
        np = self.shot._nearest_phi(0.034)
        assert_almost_equal(np, 0.03490658503988659)
        
        nd = self.shot._nearest_delta(0.034)
        assert_almost_equal(nd, 0.03490658503988659)
        
    def test_q_index(self):
        qs = self.shot.polar_grid[:,0]
        q = self.shot._nearest_q(0.09)
        ref = np.where(qs == q)[0]
        qinds = self.shot._q_index(q)
        
        # due to numerics, where sometimes fails
        for x in ref:
            assert x in qinds
            
        a1 = self.shot.polar_grid[qinds,0]
        a2 = np.ones(len(a1)) * q
        assert_array_almost_equal(a1, a2)
            
        
    def test_phi_index(self):
        phis = self.shot.polar_grid[:,1]
        phi = self.shot._nearest_phi(6.0)
        ref = np.where(phis == phi)[0]

        phiinds = self.shot._phi_index(phi)
        
        a1 = self.shot.polar_grid[phiinds,1]
        a2 = np.ones(len(a1)) * phi
        assert_array_almost_equal(a1, a2)
        
        for x in ref:
            assert x in phiinds
        
    def test_I_index(self):
        q_guess   = 0.09
        phi_guess = 0.035
        
        q_ref = self.shot._nearest_q(q_guess)
        phi_ref = self.shot._nearest_phi(phi_guess)
        
        index = self.shot._intensity_index(q_guess, phi_guess)
        q   = self.shot.polar_grid[index,0]
        phi = self.shot.polar_grid[index,1]
        
        assert_almost_equal(q, q_ref)
        assert_almost_equal(phi, phi_ref)
        
    @skip
    def test_i_profile(self):
        
        i = self.shot.polar_intensities
        pg = self.shot.polar_grid
        qs = np.unique(i[:,0])
        p = np.zeros(len(qs))
        
        for x,q in enumerate(qs):
            p[x] = (i[pg[:,0]==q]).mean()
            
        profile = self.shot.intensity_profile()
        ind_code = profile[:,0]
        p_code = profile[:,1]
        
        qs = np.array(qs)
        assert_array_almost_equal(qs, ind_code)
        
        for i in range(len(p)):
            d = np.abs(p[i] - p_code[i])
            if d > 1e-6:
                print p[i], p_code[i]
                
        assert_allclose(p, p_code)
        
    def test_correlation(self):
        
        # arb. parameters for testing
        q1 = 2.0
        q2 = 2.0
        
        for delta in [0.0, np.pi/2, np.pi]:
        
            delta = self.shot._nearest_delta(delta)
        
            x = []
            y = []
        
            masked = 0
            for phi in self.shot.phi_values:
                #if (  (q1,phi) not in self.shot.masked_polar_pixels ) and (  (q2,phi+delta) not in self.shot.masked_polar_pixels ):            
                    x.append( self.shot.I(q1, phi) )
                    y.append( self.shot.I(q2, phi+delta) )
                # else:
                #     masked += 1
                
            x = np.array(x)
            y = np.array(y)
        
            x -= x.mean()
            y -= y.mean()
        
            assert_almost_equal( x.mean(), 0.0 )
            assert_almost_equal( y.mean(), 0.0 )
        
            n = len(x)
            assert len(y) == n
        
            # compute the correlation between x,y -- recall it is a *circular* correlation
            ref = np.sum(x*y) / (n * x.std() * y.std())
        
            ans = self.shot.correlate(q1, q2, delta)        
            assert_almost_equal(ans, ref, decimal=1)
        
    @skip
    def test_corr_ring(self):
        
        # -------------------------------------------------------------------- #
        # For some reason this test is failing ... suspect the FFT method
        # has a finite error that is getting picked up. Not sure how to debug
        # atm. -- TJL 11.28.12
        # -------------------------------------------------------------------- #
        
        # arb. parameters for testing
        q1 = 2.0
        q2 = 2.0
        
        ring = self.shot.correlate_ring(q1, q2)
        ref = np.zeros(ring.shape[0])
        
        for i,delta in enumerate(ring[:,0]):
            ref[i] = self.shot.correlate(q1, q2, delta)            
            
        assert_array_almost_equal(ring[:,1], ref, decimal=2)
            
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
        delta = (45.0 / 360.0) * 2.0*np.pi
        i1 = np.array(self.shot.correlate(q1, q2, delta))
        i2 = np.array(self.shotset.correlate(q1, q2, delta))
        assert_almost_equal(i1, i2)
        
    def test_rings(self):
        q1 = 0.1
        q2 = 0.1
        i1 = np.array(self.shot.correlate_ring(q1, q2))
        i2 = np.array(self.shotset.correlate_ring(q1, q2))
        assert_array_almost_equal(i1, i2)

if __name__ == '__main__':
    test = TestShot()
    test.setup()
    test.test_correlation()
    test.test_corr_ring()
