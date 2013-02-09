
"""
Tests: src/python/xray.py
"""

import os, sys
import warnings
from nose import SkipTest

from odin import xray, utils, parse, structure
from odin.testing import skip, ref_file, expected_failure
from odin.refdata import cromer_mann_params
from mdtraj import trajectory, io

try:
    from odin import gpuscatter
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


class TestBeam(object):
    
    def setup(self):
        self.flux = 100.0
        
    def test_unit_convs(self):
        beam = xray.Beam(self.flux, energy=1.0)
        assert_allclose(beam.wavelength, 12.398, rtol=1e-3)
        assert_allclose(beam.frequency, 2.4190e17, rtol=1e-3)
        assert_allclose(beam.wavenumber, (2.0 * np.pi)/12.398, rtol=1e-3)
    
        
class TestDetector(object):
    
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
        assert_array_almost_equal(np.zeros(ref1.shape[0]), self.d.recpolar[:,1], err_msg='theta')
        assert_array_almost_equal(ref1[:,2], self.d.recpolar[:,2], err_msg='phi')
       
    def test_reciprocal_to_real(self):
        real = self.d._reciprocal_to_real(self.d.reciprocal)
        assert_array_almost_equal(real, self.d.real)
       
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
        
        
class TestFilter(object):
        
    def setup(self):
        self.d = xray.Detector.generic(spacing=0.4)
        self.i = np.abs( np.random.randn(self.d.xyz.shape[0]) )
        self.shot = xray.Shot(self.i, self.d)
        self.i_shape = self.shot.detector._grid_list[0][1][:2]        
   
    def test_generic(self):
        p_args = (0.9, self.shot.detector)
        flt = xray.ImageFilter(abs_std=3.0, polarization=p_args, border_pixels=4)
        new_i = flt(self.i, intensities_shape=self.i_shape)
        
    def test_hot(self):
        # currently just smoke test... todo : real test
        flt = xray.ImageFilter()
        flt.hot_pixels(abs_std=3.0)
        new_i = flt(self.i, intensities_shape=self.i_shape)
        
    def test_polarization(self):
        # not getting same answer as derek, todo
        P = 0.9 # arb. choice of polarization factor
        flt = xray.ImageFilter()
        flt.polarization(P, self.shot.detector)
        new_i, mask = flt(self.i, intensities_shape=self.i_shape)
        
        qxyz = self.shot.detector.reciprocal
        l = self.shot.detector.path_length
        theta_x = np.arctan( qxyz[:,0] / l )
        theta_y = np.arctan( qxyz[:,1] / l )
        f_p = 0.5 * ( (1+P) * np.power(np.cos(theta_x),2) + (1-P) * np.power(np.cos(theta_y),2) )
        ref_i = self.i * f_p
        
        assert np.all(mask == False)
        assert_allclose(new_i, ref_i)
        
    def test_detector_mask(self):
        # answer confirmed visually
        cbf = parse.CBF( ref_file('test1.cbf') )
        intensities = cbf.intensities.reshape(cbf.intensities_shape)
        flt = xray.ImageFilter()
        flt.detector_mask(border_pixels=4)
        i, mask = flt(intensities)
        ref_mask = np.load( ref_file('hist_mask_wborder.npz') )['arr_0']
        assert_array_almost_equal(mask, ref_mask)
        
    def test_histgram_segmentation(self):
        # answer confirmed visually
        cbf = parse.CBF( ref_file('test1.cbf') )
        intensities = cbf.intensities.reshape(cbf.intensities_shape)
        flt = xray.ImageFilter()
        flt._intensities_shape = cbf.intensities_shape
        mask = flt._find_detector_mask(intensities)
        ref_mask = np.load( ref_file('hist_mask.npz') )['arr_0']
        assert_array_almost_equal(mask, ref_mask)
        
    def test_borders(self):
        # answer confirmed visually
        cbf = parse.CBF( ref_file('test1.cbf') )
        intensities = cbf.intensities.reshape(cbf.intensities_shape)
        flt = xray.ImageFilter()
        flt._intensities_shape = cbf.intensities_shape
        mask = flt._find_detector_mask(intensities)
        mask = flt._mask_border(mask, 4)
        ref_mask = np.load( ref_file('hist_mask_wborder.npz') )['arr_0']
        assert_array_almost_equal(mask, ref_mask)
        
        
class TestShot(object):
        
    def setup(self):
        self.d = xray.Detector.generic(spacing=0.4)
        self.i = np.abs( np.random.randn(self.d.xyz.shape[0]) )
        self.t = trajectory.load(ref_file('ala2.pdb'))
        self.shot = xray.Shot(self.i, self.d)
        
    def test_io(self):
        if os.path.exists('test.shot'): os.system('test.shot')
        self.shot.save('test.shot')
        s = xray.Shot.load('test.shot')
        os.remove('test.shot')
        if os.path.exists('test.shot'): os.system('test.shot')        
        assert_array_almost_equal(s.intensity_profile(),
                                  self.shot.intensity_profile() )
        
    def test_sim(self):
        if not GPU: raise SkipTest
        shot = xray.Shot.simulate(self.t, 512, self.d)
        
    def test_q_values(self):
        q_max = np.max(self.shot.detector.recpolar[:,0])
        spacing = self.shot.q_spacing
        qv_ref = np.arange(spacing, q_max+spacing, spacing)
        assert_array_almost_equal(self.shot.q_values, qv_ref)
        
    def test_implicit_interpolation_smoke(self):
        s = xray.Shot(self.i, self.d)
        
    def test_unstructured_interpolation_smoke(self):
        d = xray.Detector.generic(spacing=0.4, force_explicit=True)
        s = xray.Shot(self.i, d)
        
    def test_interpolation_consistency(self):
        """ test to ensure unstructured & implicit interpolation methods
            give the same result """
        de = xray.Detector.generic(spacing=0.4, force_explicit=True)
        s1 = xray.Shot(self.i, self.d)
        s2 = xray.Shot(self.i, de)
        print "testing interpolation methods consistent"
        assert_allclose(s1.intensities, s2.intensities)
        
    def test_specific_q_interpolation(self):
        qv = [0.2, 1.0, 1.1]
        s = xray.Shot(self.i, self.d, phi_spacing=2.0, q_values=qv)
        assert s.num_q == len(qv)
        assert np.all(s.q_values == np.array(qv))
        
    def test_pgc(self):
        """ test polar_grid_as_cart() property """
        pg = self.shot.polar_grid
        pgc = self.shot.polar_grid_as_cart
        mag = np.sqrt(np.sum(np.power(pgc,2), axis=1))
        assert_array_almost_equal( mag, pg[:,0] )
        #assert_array_almost_equal( utils.arctan3(pgc[:,1], pgc[:,0]), pg[:,1] )
        maxq = self.shot.q_values.max()
        assert np.all( mag <= (maxq + 1e-6) )

    @expected_failure
    def test_pgr(self):
        """ test polar_grid_as_real_cart() property """
        
        # This test is not really working, I think there may be some numerical
        # instability in the trig functions that are used to compute this
        # function. Right now the precision below is turned way down. todo.
        
        pgr = self.shot.polar_grid_as_real_cart
        pgr_z = np.zeros((pgr.shape[0],3))
        pgr_z[:,:2] = pgr.copy()
        pgr_z[:,2] = self.shot.detector.path_length
        
        S = self.d._unit_vector(pgr_z)
        S0 = np.zeros_like(S)
        S0[:,2] = np.ones(S0.shape[0])
        pgq_z = self.shot.detector.k * (S - S0)
        pgq = pgq_z[:,:2]
        
        ref = self.shot.polar_grid_as_cart
        
        maxq = self.shot.q_values.max()
        assert np.all( np.sqrt(np.sum(np.power(ref,2), axis=1)) <= (maxq + 1e-6) )
        assert np.all( np.sqrt(np.sum(np.power(pgq,2), axis=1)) <= (maxq + 1e-6) )
        
        print "pgq:", pgq
        print "ref:", ref
        print "diff", np.sum(np.abs(pgq - ref), axis=1)
        
        assert_array_almost_equal(pgq, ref)
        
        
    def test_pgr2(self):
        """ test polar_grid_as_real_cart() property against ref implementation"""
        
        pg  = self.shot.polar_grid
        ref_pgr = np.zeros_like(pg)
        k = self.shot.detector.k
        l = self.shot.detector.path_length
        
        pg[ pg[:,0] == 0.0 ,0] = 1.0e-300
        h = l * np.tan( 2.0 * np.arcsin( pg[:,0] / (2.0*k) ) )
        
        ref_pgr[:,0] = h * np.cos( pg[:,1] )
        ref_pgr[:,1] = h * np.sin( pg[:,1] )
        
        pgr = self.shot.polar_grid_as_real_cart
        
        assert_array_almost_equal(pgr, ref_pgr)        
        
    def test_mask(self):
        """ test masking by confirming some basic stuff is reasonable """
        d = xray.Detector.generic(spacing=0.4)
        n = len(self.i)
        mask = np.zeros(n, dtype=np.bool)
        mask[ np.random.random_integers(0, n, 10) ] = np.bool(True)
        s = xray.Shot(self.i, d, mask=mask)
        ip = s.intensity_profile()        
        ref_mean = np.mean(self.i[np.bool(True)-mask])
        assert_allclose( ref_mean, s.intensities.mean(), rtol=1e-04 )
        assert_allclose( ref_mean, s.polar_intensities.mean(), rtol=0.25 )
        
    def test_mask2(self):
        """ test masking by ref. implementation """
        
        n = len(self.i)
        mask = np.zeros(n, dtype=np.bool)
        mask[ np.random.random_integers(0, n, 10) ] = np.bool(True)
        
        s = xray.Shot(self.i, self.d, mask=mask)
        
        polar_mask = np.zeros(s.num_datapoints, dtype=np.bool)
        xyz = s.detector.reciprocal
        pgc = s.polar_grid_as_cart
        
        # the max distance, r^2 - a factor of 10 helps get the spacing right
        r2 =  np.sum( np.power( xyz[0] - xyz[1], 2 ) ) * 10.0
        masked_cart_points = xyz[mask,:2]
        
        ref_polar_mask = np.zeros(s.num_datapoints, dtype=np.bool)        
        for mp in masked_cart_points:
            d2 = np.sum( np.power( mp - pgc[:,:2], 2 ), axis=1 )
            ref_polar_mask[ d2 < r2 ] = np.bool(True)
        
        assert_array_equal(ref_polar_mask, s.polar_mask)
        
    def test_nearests(self):
        
        print "q", self.shot.q_values
        print "phi", self.shot.phi_values
        
        nq = self.shot._nearest_q(0.019)
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
        
    def test_i_profile(self):
        
        i = self.shot.polar_intensities
        pg = self.shot.polar_grid
        qs = self.shot.q_values
        p = np.zeros(len(qs))
        
        for x,q in enumerate(qs):
            p[x] = (i[pg[:,0]==q]).mean()
            
        profile = self.shot.intensity_profile()
        ind_code = profile[:,0]
        p_code = profile[:,1]
        
        qs = np.array(qs)
        assert_array_almost_equal(qs, ind_code)                
        assert_allclose(p, p_code, rtol=1)
    
    def test_correlation(self):
        
        # arb. parameters for testing
        q1 = 2.0
        q2 = 2.0
        
        x = self.shot.I_ring(q1)
        y = self.shot.I_ring(q2)
        assert len(x) == len(y)
        
        x -= x.mean()
        y -= y.mean()
        
        ref = np.zeros(len(x))
        for i in range(len(x)):
            ref[i] = np.correlate(x, np.roll(y, i))
        ref /= ref[0]
                
        for i in range(10):
            delta = self.shot.phi_values[i]
            ans = self.shot.correlate(q1, q2, delta)
            assert_almost_equal(ans, ref[i], decimal=1)
        
    def test_corr_ring(self):
                
        # arb. parameters for testing
        q1 = 2.0
        q2 = 2.0
        
        ring = self.shot.correlate_ring(q1, q2)
        
        # reference computation
        x = self.shot.I_ring(q1)
        y = self.shot.I_ring(q2)
        assert len(x) == len(y)
        
        x -= x.mean()
        y -= y.mean()
        
        ref = np.zeros(len(x))
        for i in range(len(x)):
            ref[i] = np.correlate(x, np.roll(y, i))
        ref /= ref[0]
                    
        assert_allclose(ref, ring[:,1])
     
    def test_simulate_cpu_only(self):
        d = xray.Detector.generic(spacing=0.6)
        x = xray.Shot.simulate(self.t, 1, d)
        
    def test_simulate_gpu_only(self):
        if not GPU: raise SkipTest
        d = xray.Detector.generic(spacing=0.6)
        x = xray.Shot.simulate(self.t, 512, d)
            
    def test_simulate_gpu_and_cpu(self):
        if not GPU: raise SkipTest
        d = xray.Detector.generic(spacing=0.6)
        x = xray.Shot.simulate(self.t, 513, d)
        
        
class TestShotset():
    
    def setup(self):
        self.shot = xray.Shot.load(ref_file('refshot.shot'))
        self.shotset = xray.Shotset([self.shot])
        self.t = trajectory.load(ref_file('ala2.pdb'))
        
    def test_iter_n_slice(self):
        s = self.shotset[0]
        for s in self.shotset: print s
        
    def test_add_n_len(self):
        ss = self.shotset + self.shotset
        assert len(ss) == 2
        
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
    
        
class TestCorrelationCollection(object):
    
    def setup(self):
        self.shot = xray.Shot.load(ref_file('refshot.shot'))
        self.cc = xray.CorrelationCollection(self.shot)
        
    def test_ring(self):
        q1 = self.cc.q_values[0]
        q2 = self.cc.q_values[0]
        ring1 = self.cc.ring(q1, q2)
        ring2 = self.shot.correlate_ring(q1, q2)[:,1]
        assert_allclose(ring1, ring2)
        
    def test_coefficients_smoke(self):
        cl = self.cc.legendre_coeffecients()
        
    def test_coefficients(self):
        order = 2000
        cc = self.cc

        # inject a simple test case into the cc
        cc.q_values = np.array([1.0])
        cc.num_q = 1

        # compute the values of psi to use
        t1 = np.arctan( 1.0 / (2.0*cc.k) )
        t2 = np.arctan( 1.0 / (2.0*cc.k) )
        psi = np.arccos( np.cos(t1)*np.cos(t2) + np.sin(t1)*np.sin(t2) \
                         * np.cos( cc.phi_values * 2. * np.pi/float(cc.num_phi) ) )

        # inject a cosine function as the correlation
        correlation = np.cos( psi * 1000 )
        cc._correlation_data = { (1.0, 1.0) : correlation }

        # compute the coefficients and re-construct the cos function
        cl = cc.legendre_coeffecients(order=order)
        cl = cl[:,0,0]
        c = np.zeros( 2 * cl.shape[0] )
        c[::2] = cl
        pred = np.polynomial.legendre.legval( np.cos(psi), c)

        assert_allclose(pred, correlation, rtol=1e-01)
        
