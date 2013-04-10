
"""
Tests: src/python/xray.py
"""

import os, sys
import warnings
from nose import SkipTest

from odin import xray, utils, parse, structure, math2, utils
from odin.testing import skip, ref_file, expected_failure, brute_force_masked_correlation
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
        
    def test_num_pixels(self):
        assert self.bg.num_pixels == np.product(self.shape)
        
    def test_grid_corners(self):
        c = self.bg.get_grid_corners(0)
        assert_array_almost_equal(c[0,:], self.p)
        assert_array_almost_equal(c[2,:], np.array([1.0*10, 0.0, 1.0])) # slow
        assert_array_almost_equal(c[1,:], np.array([0.0, 2.0*10, 1.0])) # fast
        assert_array_almost_equal(c[3,:], np.array([1.0*10, 2.0*10, 1.0]))
        
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
                                       
    def test_implicit_to_explicit(self):
        xyz_imp = self.d.real
        self.d.implicit_to_explicit()
        assert_array_almost_equal(xyz_imp, self.d.real)
        
    def test_evaluate_qmag(self):
        # doubles as a test for _evaluate_theta
        x = np.zeros((5, 3))
        x[:,0] = np.random.randn(5)
        x[:,2] = self.l
        
        S = x.copy()
        S = S / np.sqrt( np.sum( np.power(S, 2), axis=1 ) )[:,None]
        S -= self.d.beam_vector
        
        b = xray.Beam(1, energy=self.energy)
        qref = b.k * np.sqrt( np.sum( np.power(S, 2), axis=1 ) )
        
        qmag = self.d.evaluate_qmag(x)
        assert_allclose(qref, qmag)
        
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
        
        # |q| = k*sqrt{ 2 - 2 cos(theta) }
        ref1[:,0] = self.d.k * np.sqrt( 2.0 - 2.0 * np.cos(self.d.polar[:,1]) )
        
        # q_theta = theta / 2 (one-theta convention)
        ref1[:,1] = self.d.polar[:,1] / 2.0 # not working atm
        
        # q_phi is the same as polar
        ref1[:,2] = self.d.polar[:,2].copy()
        
        assert_array_almost_equal(ref1[:,0], self.d.recpolar[:,0], err_msg='|q|')
        assert_array_almost_equal(ref1[:,1], self.d.recpolar[:,1], err_msg='theta')
        assert_array_almost_equal(ref1[:,2], self.d.recpolar[:,2], err_msg='phi')
        
    def test_compute_intersect(self):
        
        # build a simple grid and turn it into a detector
        bg = xray.BasisGrid()
        p = np.array([0.0, 0.0, 1.0])
        s = np.array([1.0, 0.0, 0.0])
        f = np.array([0.0, 1.0, 0.0])
        shape = (10, 10)
        bg.add_grid(p, s, f, shape)
        d = xray.Detector(bg, 2.0*np.pi/1.4)
        
        # compute a set of q-vectors corresponding to a slightly offset grid
        xyz_grid = bg.to_explicit()
        xyz_off = xyz_grid.copy()
        xyz_off[:,0] += 0.5
        xyz_off[:,1] += 0.5
        q_vectors = d._real_to_reciprocal(xyz_off)
        
        # b/c s/f vectors are unit vectors, where they intersect s/f is simply
        # their coordinates. The last row and column will miss, however        
        intersect_ref = np.logical_and( (xyz_off[:,0] <= 9.0),
                                        (xyz_off[:,1] <= 9.0) )
                                        
        pix_ref = xyz_off[intersect_ref,:2]
        
        # compute the intersection from code
        pix, intersect = d._compute_intersections(q_vectors, 0) # 0 --> grid_index
        print pix, intersect
        
        assert_array_almost_equal(intersect_ref, intersect)
        assert_array_almost_equal(pix_ref, pix)
        
    def test_serialization(self):
        s = self.d._to_serial()
        d2 = xray.Detector._from_serial(s)
        assert_array_almost_equal(d2.xyz, self.d.xyz)
        
    def test_io(self):
        if os.path.exists('r.dtc'): os.system('rm r.dtc')
        self.d.save('r.dtc')
        d = xray.Detector.load('r.dtc')
        if os.path.exists('r.dtc'): os.system('rm r.dtc') 
        assert_array_almost_equal(d.xyz, self.d.xyz)
        
    def test_q_max(self):
        ref_q_max = np.max(self.d.recpolar[:,0])
        assert_almost_equal(self.d.q_max, ref_q_max, decimal=2)
        
        
class TestFilter(object):
        
    def setup(self):
        self.d = xray.Detector.generic(spacing=0.4)
        self.i = np.abs( np.random.randn(self.d.xyz.shape[0]) )
        self.shot = xray.Shot(self.i, self.d)
        self.i_shape = self.d.xyz.shape[0]
        
    @skip
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
        
    @skip
    def test_detector_mask(self):
        # answer confirmed visually
        cbf = parse.CBF( ref_file('test1.cbf') )
        intensities = cbf.intensities.reshape(cbf.intensities_shape)
        flt = xray.ImageFilter()
        flt.detector_mask(border_pixels=4)
        i, mask = flt(intensities)
        ref_mask = np.load( ref_file('hist_mask_wborder.npz') )['arr_0']
        assert_array_almost_equal(mask, ref_mask)
        
    def test_histogram_segmentation(self):
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
        self.l = 50.0
        self.d = xray.Detector.generic(spacing=0.4, l=self.l)
        self.i = np.abs( np.random.randn(self.d.xyz.shape[0]) )
        self.t = trajectory.load(ref_file('ala2.pdb'))
        self.shot = xray.Shot(self.i, self.d)
        
    def test_mask_argument(self):
        # simple smoke test to make sure we can interpolate with a mask
        q_value = np.array([1.0, 2.0])
        num_phi = 360
        mask = np.random.binomial(1, 0.1, size=self.i.shape)
        s = xray.Shot(self.i, self.d, mask=mask)
        s._interpolate_to_polar()
        
    # missing test: test_assemble (ok for now)
        
    def test_polar_grid(self):
        pg = self.shot.polar_grid([1.0], 360)
        pg_ref = np.zeros((360, 2))
        pg_ref[:,0] = 1.0
        pg_ref[:,1] = np.linspace(0.0, 2.0*np.pi, num=360)
        assert_array_almost_equal(pg, pg_ref)
        
    def test_polar_grid_as_cart(self):
        pg = self.shot.polar_grid(self.q_values, self.num_phi)
        pgc = self.shot.polar_grid_as_cart(self.q_values, self.num_phi)
        mag = np.sqrt(np.sum(np.power(pgc,2), axis=1))
        assert_array_almost_equal( mag, pg[:,0] )
        maxq = self.q_values.max()
        assert np.all( mag <= (maxq + 1e-6) )
        
    def test_interpolate_to_polar(self):
        # doubles as a test for _implicit_interpolation
        q_values = np.array([2.4, 2.67, 3.0]) # should be a peak at |q|=2.67
        t = structure.load_coor(ref_file('gold1k.coor'))
        s = xray.Shot.simulate(t, 3, self.d)
        pi, pm = s._interpolate_to_polar(q_values=q_values)
        pi = pi.reshape(360,len(q_values))
        ip = pi.sum(1) # should be a peak at |q|=2.67
        assert ip[1] > ip[0]
        assert ip[1] > ip[2]
        
    def test_explicit_interpolation(self):
        # doubles as a test for _implicit_interpolation
        q_values = np.array([2.4, 2.67, 3.0]) # should be a peak at |q|=2.67
        t = structure.load_coor(ref_file('gold1k.coor'))
        self.d.implicit_to_explicit()
        s = xray.Shot.simulate(t, 3, self.d)
        pi, pm = s._interpolate_to_polar(q_values=q_values)
        pi = pi.reshape(360,len(q_values))
        ip = pi.sum(0) # should be a peak at |q|=2.67
        assert ip[1] > ip[0]
        assert ip[1] > ip[2]
        
        assert False # there is a bug somewhere... (visual)
        
    def test_interpolation_consistency(self):
        q_values = np.array([0.2, 0.4])
        de = xray.Detector.generic(spacing=0.4, force_explicit=True)
        s1 = xray.Shot(self.i, self.d)
        s2 = xray.Shot(self.i, de)
        p1, m1 = s1._interpolate_to_polar(q_values=q_values)
        p2, m2 = s2._interpolate_to_polar(q_values=q_values)
        assert_allclose(p1, p2, err_msg='interp intensities dont match')
        assert_allclose(m1, m2, err_msg='polar masks dont match')
        
    def test_i_profile(self):
        # doubles as a test for intensity_maxima()
        t = structure.load_coor(ref_file('gold1k.coor'))
        s = xray.Shot.simulate(t, 5, self.d)
        p = s.intensity_profile()
        m = s.intensity_maxima()
        assert np.any(np.abs(p[m,0] - 2.67) < 1e-1) # |q| = 2.67 is in {maxima}
        
    def test_rotated_beam(self):
        raise NotImplementedError('test not in')
                
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
        q_values = np.linspace(0.1, 0.5, num=11)
        rings = self.shot.to_rings(q_values)
        rings_ip = rings.intensity_profile()
        s_bins = np.linspace(0.02, 0.5, num=13)
        shot_ip = self.shot.intensity_profile(n_bins=s_bins)
        assert_allclose(rings_ip[:,0], shot_ip[:,0], err_msg='test impl error')
        assert_allclose(rings_ip, shot_ip, err_msg='intensity mismatch')
        
    def test_io(self):
        if os.path.exists('test.shot'): os.remove('test.shot')
        self.shot.save('test.shot')
        s = xray.Shot.load('test.shot')
        if os.path.exists('test.shot'): os.remove('test.shot')
        assert_array_almost_equal(s.intensity_profile(),
                                  self.shot.intensity_profile() )
        
        
class TestShotset():
    
    # note : Save, load, and to_rings methods are currently just being tested
    #        via the tests in TestShot above. Should add these here later.
    
    def setup(self):
        self.d = xray.Detector.generic(spacing=0.4)
        self.t = trajectory.load(ref_file('ala2.pdb'))
        self.shot = xray.Shot.simulate(self.t, 1, self.d)
        self.shotset = xray.Shotset([self.shot])
        
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
        
        # make sure err is not thrown when detectors are the same object
        d1 = xray.Detector.generic(spacing=0.4)
        s1 = xray.Shot.simulate(self.t, 1, d1)
        s2 = xray.Shot.simulate(self.t, 1, d1)
        ss = xray.Shotset([s1, s2])
        
        # make sure err is not thrown when detectors have the same pixels
        d2 = xray.Detector.generic(spacing=0.4)
        s3 = xray.Shot.simulate(self.t, 1, d2)
        ss = xray.Shotset([s1, s3])
        
        # make sure err *is* thrown when detectors are different
        d3 = xray.Detector.generic(spacing=0.5)
        s4 = xray.Shot.simulate(self.t, 1, d3)
        try:
            ss = xray.Shotset([s1, s4])
            raise RuntimeError('detector check should have thrown err')
        except AttributeError as e:
            print e
        
    def test_mask_checking(self):
        
        d  = xray.Detector.generic(spacing=0.4)
        n_pixels = d.xyz.shape[0]
        i  = np.abs( np.random.randn(n_pixels) )
        
        # make some masks
        m1 = np.random.binomial(1, 0.5, n_pixels)
        m2 = m1
        m3 = m1.copy()
        m4 = np.random.binomial(1, 0.5, n_pixels)
        
        # make sure err is not thrown when masks are the same object
        s1 = xray.Shot(i, d, mask=m1)
        s2 = xray.Shot(i, d, mask=m2)
        ss = xray.Shotset([s1, s2])
        
        # make sure err is not thrown when masks have the same pixels
        s1 = xray.Shot(i, d, mask=m1)
        s3 = xray.Shot(i, d, mask=m3)
        ss = xray.Shotset([s1, s3])
        
        # make sure err *is* thrown when detectors are different
        s1 = xray.Shot(i, d, mask=m1)
        s4 = xray.Shot(i, d, mask=m4)
        try:
            ss = xray.Shotset([s1, s4])
            raise RuntimeError('mask check should have thrown err')
        except AttributeError as e:
            print e
        
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
        rings = xray.Rings.simulate(self.traj, 1, self.q_values, 
                                    self.num_phi, 1) # 1 molec, 1 shots
        # todo: better than smoke test
                                      
    def test_q_index(self):
        assert self.rings.q_index(1.0) == 0
        assert self.rings.q_index(2.0) == 1
        
    def test_intensity_profile(self):
        q_values = [2.4, 2.67, 3.0] # should be a peak at |q|=2.67
        t = structure.load_coor(ref_file('gold1k.coor'))
        rings = xray.Rings.simulate(t, 10, q_values, self.num_phi, 1) # 3 molec, 1 shots
        ip = rings.intensity_profile()
        assert ip[1,1] > ip[0,1]
        assert ip[1,1] > ip[2,1]
        
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
        ref /= ref[0] # NORMALIZE
        
        for delta in range(10):
            ans = self.rings.correlate(q, q, delta)
            assert_almost_equal(ans, ref[delta], decimal=1)
            
    def test_brute_correlation_wo_mask(self):
        # this might be better somewhere else, but is here for now
        x = np.random.randn(100) + 0.1 * np.arange(100)
        x_bar = np.mean(x)
        x -= x_bar
        mask = np.ones(100, dtype=np.bool)
        
        c = brute_force_masked_correlation(x, mask)
        ref = np.zeros(100)
        
        for i in range(100):
            ref[i] = np.correlate(x, np.roll(x, i))
        ref /= ref[0]
            
        assert_allclose(ref, c)
            
    def test_correlation_w_mask(self):
        # brute force compute the correlator, skipping over gaps
        
        # set up a rings obj with a mask
        rings = xray.Rings.simulate(self.traj, 1, self.q_values, self.num_phi, 1)
        polar_intensities = rings.polar_intensities
        polar_mask = np.random.binomial(1, 0.75, size=polar_intensities.shape).astype(np.bool)
        rings = xray.Rings(self.q_values, polar_intensities, self.rings.k,
                           polar_mask=polar_mask)
        
        # compute a reference autocorrelator for the first ring
        pm = polar_mask[0,0,:].flatten()
        x = polar_intensities[0,0,:].flatten()
        ref_corr = brute_force_masked_correlation(x, pm)
        
        # compute the correlator the usual way & compare
        corr = np.zeros(rings.num_phi)
        for delta in range(rings.num_phi):
            corr[delta] = rings._correlate_by_index(0, 0, delta)
        
        assert_allclose(ref_corr, corr, rtol=1e-03)
        
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
        # set up a rings obj with a mask
        rings = xray.Rings.simulate(self.traj, 1, self.q_values, self.num_phi, 1)
        polar_intensities = rings.polar_intensities
        polar_mask = np.random.binomial(1, 0.75, size=polar_intensities.shape).astype(np.bool)
        rings = xray.Rings(self.q_values, polar_intensities, self.rings.k,
                           polar_mask=polar_mask)
        
        # compute a reference autocorrelator for the first ring
        pm = polar_mask[0,0,:].flatten()
        x = rings.polar_intensities[0,0,:].flatten()
        ref_corr = brute_force_masked_correlation(x, pm)
        
        # compute the correlator the usual way & compare
        corr = rings._correlate_ring_by_index(0, 0)
        
        assert_allclose(ref_corr, corr, rtol=1e-03)
        
    def test_correlate_inters(self):
        # inject a specific and carefully engineered test case into a Rings obj
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
        os.remove('test.ring')
        assert np.all( self.rings.polar_intensities == r.polar_intensities)
        

class TestMisc(object):
    
    def test_q_values(self):
        
        q_values = np.array([1.0, 2.0, 3.0])
        num_phi = 360
        k = 2.0 * np.pi / 1.4
        
        qxyz = xray._q_grid_as_xyz(q_values, num_phi, k)
        
        # assert that the above vectors are the correct length
        assert np.all( np.abs( np.sqrt( np.sum( np.power(qxyz,2), axis=1 ) ) - \
                               np.repeat(q_values, num_phi)) < 1e-6 )
                               
    def test_iprofile_consistency(self):
        
        t = structure.load_coor(ref_file('gold1k.coor'))
        d = xray.Detector.generic()
        s = xray.Shot.simulate(t, 5, d)
        
        # compute from polar interp
        pi, pm = s._implicit_interpolation(q_values, num_phi)
        pi = pi.reshape(len(q_values), num_phi)
        ip1 = np.zeros(len(q_values), 2)
        ip1[:,1] = q_values
        ip1[:,1] = pi.sum(1)
        
        # compute from detector
        ip2 = s.intensity_profile()

        # compute from rings
        r = xray.Rings.simulate(t, 5, q_values, 360, 1) 
        ip3 = r.intensity_profile()
                               
        # make sure maxima are all similar
        m1 = ip1[utils.maxima(ip1[:,1]),0]
        m2 = ip2[utils.maxima(ip2[:,1]),0]
        m3 = ip3[utils.maxima(ip3[:,1]),0]
        
        assert_allclose(m1, m2)
        assert_allclose(m1, m3)
        assert_allclose(m2, m3)
        