
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
        self.n_photons = 100.0
        
    def test_unit_convs(self):
        beam = xray.Beam(self.n_photons, energy=1.0)
        assert_allclose(beam.wavelength, 12.398, rtol=1e-3)
        assert_allclose(beam.frequency, 2.4190e17, rtol=1e-3)
        assert_allclose(beam.wavenumber, (2.0 * np.pi)/12.398, rtol=1e-3)
    
        
class TestBasisGrid(object):
    
    def setup(self):
        self.p = np.array([0.0, 0.0, 1.0])
        self.s = np.array([1.0, 0.0, 0.0])
        self.f = np.array([0.0, 2.0, 0.0])
        self.shape = (10, 10)
        self.grid_list = [(self.p, self.s, self.f, self.shape)]
        self.bg = xray.BasisGrid(self.grid_list)
        
    def test_add_grid(self):
        nbg = xray.BasisGrid()
        nbg.add_grid(*self.grid_list[0])
        assert_array_almost_equal(nbg.to_explicit(), self.bg.to_explicit())
    
    def test_add_using_center(self):
        center = np.array([4.5, 9, 1.0])
        nbg = xray.BasisGrid()
        nbg.add_grid_using_center(center, self.s, self.f, self.shape)
        assert_array_almost_equal(nbg.to_explicit(), self.bg.to_explicit())
    
    def test_get_grid(self):
        assert self.bg.get_grid(0) == self.grid_list[0]
        
    def test_to_explicit(self):
        ref = np.zeros((100,3))
        mg = np.mgrid[0:9:10j,0:18:10j]
        ref[:,0] = mg[0].flatten()
        ref[:,1] = mg[1].flatten()
        ref[:,2] = 1.0
        assert_array_almost_equal(self.bg.to_explicit(), ref)
        
    def test_grid_as_explicit(self):
        ref = np.zeros((10,10,3))
        mg = np.mgrid[0:9:10j,0:18:10j]
        ref[:,:,0] = mg[0]
        ref[:,:,1] = mg[1]
        ref[:,:,2] = 1.0
        assert_array_almost_equal(self.bg.grid_as_explicit(0), ref)
    
        
class TestDetector(object):
    
    def setup(self):
        self.spacing   = 0.05
        self.lim       = 10.0
        self.energy    = 0.7293
        self.n_photons = 100.0
        self.l         = 50.0
        self.d = xray.Detector.generic(spacing = self.spacing,
                                       lim = self.lim,
                                       energy = self.energy,
                                       photons_scattered_per_shot = self.n_photons,
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
        beam = xray.Beam(self.n_photons, energy=self.energy)
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
        
    @skip
    def test_polarization(self):
        
        # todo : need to figure out correct formula
        
        P = 0.9 # arb. choice of polarization factor
        
        flt = xray.ImageFilter()
        flt.polarization(P, self.shot.detector)
        new_i, mask = flt(self.i, intensities_shape=self.i_shape)
        
        thetas = self.shot.detector.polar[:,1].copy() / 2.0 # this is the crystallographic theta
        phis   = self.shot.detector.polar[:,2].copy()
        
        cf  = ( 1.0 + P ) * ( 1.0 - np.power( np.sin(thetas) * np.cos(phis), 2 ))
        cf += ( 1.0 - P ) * ( 1.0 - np.power( np.sin(thetas) * np.sin(phis), 2 ))
        ref_i = self.i / cf
             
        # all that really matters is that the ratio of the intensities is the
        # same for all pixels -- intensity units are arbitrary
        ratio = new_i / ref_i
        
        import matplotlib.pyplot as plt
        plt.hist(ratio,100)
        plt.show()
        
        assert np.all(mask == False)
        assert_allclose(ratio - ratio.mean(), np.zeros(len(ratio)))
        
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
        self.q_values = np.array([1.0, 2.0])
        self.num_phi  = 360
        self.d = xray.Detector.generic(spacing=0.4)
        self.i = np.abs( np.random.randn(self.d.xyz.shape[0]) )
        self.t = trajectory.load(ref_file('ala2.pdb'))
        self.shot = xray.Shot(self.i, self.d)
        
    def test_io(self):
        if os.path.exists('test.shot'): os.remove('test.shot')
        self.shot.save('test.shot')
        s = xray.Shot.load('test.shot')
        if os.path.exists('test.shot'): os.remove('test.shot')
        assert_array_almost_equal(s.intensity_profile(),
                                  self.shot.intensity_profile() )
    
    def test_assemble(self):
        raise NotImplementedError('test not in')
        
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
        
    def test_pgc(self):
        """ test polar_grid_as_cart() property """
        pg = self.shot.polar_grid(self.q_values, self.num_phi)
        pgc = self.shot.polar_grid_as_cart(self.q_values, self.num_phi)
        mag = np.sqrt(np.sum(np.power(pgc,2), axis=1))
        assert_array_almost_equal( mag, pg[:,0] )
        maxq = self.q_values.max()
        assert np.all( mag <= (maxq + 1e-6) )

    @skip
    def test_pgr(self):
        """ test polar_grid_as_real_cart() property """
        
        # This test is not really working, I think there may be some numerical
        # instability in the trig functions that are used to compute this
        # function. Right now the precision below is turned way down. todo.
        
        pgr = self.shot.polar_grid_as_real_cart(self.q_values, self.num_phi)
        pgr_z = np.zeros((pgr.shape[0],3))
        pgr_z[:,:2] = pgr.copy()
        pgr_z[:,2] = self.shot.detector.path_length
        
        S = self.d._unit_vector(pgr_z)
        S0 = np.zeros_like(S)
        S0[:,2] = np.ones(S0.shape[0])
        pgq_z = self.shot.detector.k * (S - S0)
        pgq = pgq_z[:,:2]
        
        ref = self.shot.polar_grid_as_cart(self.q_values, self.num_phi)
        
        maxq = self.shot.q_values.max()
        assert np.all( np.sqrt(np.sum(np.power(ref,2), axis=1)) <= (maxq + 1e-6) )
        assert np.all( np.sqrt(np.sum(np.power(pgq,2), axis=1)) <= (maxq + 1e-6) )
        
        print "pgq:", pgq
        print "ref:", ref
        print "diff", np.sum(np.abs(pgq - ref), axis=1)
        
        assert_array_almost_equal(pgq, ref)
        
        
    def test_pgr2(self):
        """ test polar_grid_as_real_cart() property against ref implementation"""
        
        pg  = self.shot.polar_grid(self.q_values, self.num_phi)
        ref_pgr = np.zeros_like(pg)
        k = self.shot.detector.k
        l = self.shot.detector.path_length
        
        pg[ pg[:,0] == 0.0 ,0] = 1.0e-300
        h = l * np.tan( 2.0 * np.arcsin( pg[:,0] / (2.0*k) ) )
        
        ref_pgr[:,0] = h * np.cos( pg[:,1] )
        ref_pgr[:,1] = h * np.sin( pg[:,1] )
        
        pgr = self.shot.polar_grid_as_real_cart(self.q_values, self.num_phi)
        
        assert_array_almost_equal(pgr, ref_pgr)
        
    def test_overlap(self):
        
        # test is two squares, offset
        xy1 = np.mgrid[0:2:3j,0:2:3j].reshape(9,2)
        xy2 = np.mgrid[0:2:2j,0:2:2j].reshape(4,2) - 0.5
        overlap = self.shot._overlap(xy1, xy2)
        
        ref = np.array([0, 1, 2, 6]) # by hand
        
        assert_array_almost_equal(ref, overlap)
        
    def test_overlap_implicit(self):
        pass # todo : waiting for new basis grid method (fxn call change)
        
    def test_real_mask_to_polar(self):
        
        # mask the top-left real corner, make sure pixels in [0,pi/2] are masked
        # while others aren't
        
        q_values = np.array([1.0, 2.0])
        num_phi = 360
        
        mask = np.ones(self.i.shape, dtype=np.bool)
        mask[ (self.d.xyz[:,0] > 0.0) * (self.d.xyz[:,1] > 0.0) ] = np.bool(False)
        
        s = xray.Shot(self.i, self.d, mask=mask)
        polar_mask = s._real_mask_to_polar(q_values, num_phi)
        polar_grid = s.polar_grid(q_values, num_phi)
        
        print "polar num *not* masked, total:", np.sum(polar_mask), polar_mask.shape[0]
        
        # make sure quadrant I is masked
        QI = (polar_grid[:,1] > 0.0) * (polar_grid[:,1] < (np.pi/2.0))
        assert np.all( polar_mask[QI] == np.bool(False) )
        
        # make sure quadrant III is not masked
        QIII = (polar_grid[:,1] > 3.17) * (polar_grid[:,1] < 4.6)
        assert np.sum( np.logical_not(polar_mask[QIII]) ) == 0.0
        
    def test_mask_argument(self):
        # simple smoke test to make sure we can interpolate with a mask
        q_value = np.array([1.0, 2.0])
        num_phi = 360
        mask = np.random.binomial(1, 0.1, size=self.i.shape)
        s = xray.Shot(self.i, self.d, mask=mask)
        s._interpolate_to_polar()
        
    def test_i_profile(self):
        t = structure.load_coor(ref_file('gold1k.coor'))
        s = xray.Shot.simulate(t, 1, self.d)
        p = s.intensity_profile()
        m = s.intensity_maxima()
        assert np.abs(p[m[0],0] - 2.6763881) < 1e-3
                
    def test_sim(self):
        if not GPU: raise SkipTest
        shot = xray.Shot.simulate(self.t, 512, self.d)
     
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
        
    def test_to_rings(self):
        raise NotImplementedError('test not in')
        
        
class TestShotset():
    
    # note : Save, load, and to_rings methods are currently just being tested
    #        via the tests in TestShot above. Should add these here later.
    
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
        raise NotImplementedError('test not in')
        
    def test_profiles(self):
        i1 = self.shot.intensity_profile()
        i2 = self.shotset.intensity_profile()
        assert_array_almost_equal(i1, i2)
    
        
class TestRings(object):
    
    def setup(self):
        self.q_values = np.array([1.0, 2.0])
        self.num_phi  = 360
        self.traj     = trajectory.load(ref_file('ala2.pdb'))
        self.rings    = xray.Rings.simulate(self.traj, 1, self.q_values, 
                                            self.num_phi, 2) # 1 molec, 2 shots
                              
    def test_sanity(self):
        assert self.rings.polar_intensities.shape == (2, len(self.q_values), self.num_phi)
                                            
    def test_simulation(self):
        
        raise NotImplementedError('test not in')
                                      
                                      
    def test_q_index(self):
        assert self.rings.q_index(1.0) == 0
        assert self.rings.q_index(2.0) == 1
        
    def test_intensity_profile(self):
        raise NotImplementedError('test not in')
        
    def test_correlation(self):
        
        q = 1.0 # chosen arb.
        q_ind = self.rings.q_index(q)
        
        x = self.rings.polar_intensities[0,q_ind,:].flatten()
        y = self.rings.polar_intensities[0,q_ind,:].flatten()
        assert len(x) == len(y)
        
        x -= x.mean()
        y -= y.mean()
        
        ref = np.zeros(len(x))
        for i in range(len(x)):
            ref[i] = np.correlate(x, np.roll(y, i))
        ref /= ref[0]
        
        for i in range(10):
            delta = i
            ans = self.rings.correlate(q, q, delta)
            assert_almost_equal(ans, ref[i], decimal=1)
            
    def test_correlation_w_mask(self):
        raise NotImplementedError('test not in')

    def test_corr_ring(self):

        q1 = 1.0 # chosen arb.
        q_ind = self.rings.q_index(q1)

        ring = self.rings.correlate_ring(q1, q1)

        # reference computation
        x = self.rings.polar_intensities[0,q_ind,:].flatten()
        y = self.rings.polar_intensities[0,q_ind,:].flatten()
        assert len(x) == len(y)

        x -= x.mean()
        y -= y.mean()

        ref = np.zeros(len(x))
        for i in range(len(x)):
            ref[i] = np.correlate(x, np.roll(y, i))
        ref /= ref[0]

        assert_allclose(ref, ring)
        
    def test_corr_ring_w_mask(self):
        raise NotImplementedError('test not in')
        
    def test_coefficients_smoke(self):
        order = 6
        cl = self.rings.legendre(order)
        assert cl.shape == (order, self.rings.num_q, self.rings.num_q)
        
    def test_coefficients(self):
        
        order = 10000
        q1 = 0
        q2 = 0
        cl = self.rings.legendre(order)[:,q1,q1] # keep only q1, q1 correlation
        assert len(cl) == order

        # compute the values of psi to use
        t1 = np.arctan( self.rings.q_values[q1] / (2.0*self.rings.k) )
        t2 = np.arctan( self.rings.q_values[q2] / (2.0*self.rings.k) )
        psi = np.arccos( np.cos(t1)*np.cos(t2) + np.sin(t1)*np.sin(t2) \
                         * np.cos( self.rings.phi_values * 2. * \
                         np.pi/float(self.rings.num_phi) ) )

        # reconstruct the correlation function
        pred = np.polynomial.legendre.legval(np.cos(psi), cl)
        
        # make sure it matches up with the raw correlation
        ring = self.rings._correlate_ring_by_index(q1, q2)

        # this is a high tol, but it's workin --TJL
        assert_allclose(pred, ring, rtol=0.25)
        
    def test_io(self):
        self.rings.save('test.ring')
        r = xray.Rings.load('test.ring')
        assert np.all( self.rings.polar_intensities == r.polar_intensities)
        
