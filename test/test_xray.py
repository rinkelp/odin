
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
        
    @expected_failure
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
        #self.shot = xray.Shot.load(ref_file('refshot.shot'))
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
        
        assert_array_almost_equal(pgq, ref, decimal=1)
        
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
    
    @expected_failure
    def test_correlation(self):
        
        # todo
        
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
            assert_almost_equal(ans, ref, decimal=6)
        
    @expected_failure
    def test_corr_ring(self):
                
        # -------------------------------------------------------------------- #
        # For some reason this test is failing ... suspect the FFT method
        # has a finite error that is getting picked up. Not sure how to debug
        # atm. todo. -- TJL 11.28.12
        # -------------------------------------------------------------------- #
        
        # arb. parameters for testing
        q1 = 2.0
        q2 = 2.0
        
        ring = self.shot.correlate_ring(q1, q2)
        ref = np.zeros(ring.shape[0])
        
        for i,delta in enumerate(ring[:,0]):
            ref[i] = self.shot.correlate(q1, q2, delta)            
            
        assert_array_almost_equal(ring[:,1], ref, decimal=2)
     
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
        
        
class TestDebye(object):
    
    @skip
    def test_against_reference_implementation(self):
        
        def debye_reference(trajectory, weights=None, q_values=None):
            """
            Computes the Debye scattering equation for the structures in `trajectory`,
            producing the theoretical intensity profile.

            Treats the object `trajectory` as a sample from a Boltzmann ensemble, and
            averages the profile from each snapshot in the trajectory. If `weights` is
            provided, weights the ensemble accordingly -- otherwise, all snapshots are
            given equal weights.
            
            THIS IS A PYTHON REFERENCE IMPLEMENTATAION.

            Parameters
            ----------
            trajectory : mdtraj.trajectory
                A trajectory object representing a Boltzmann ensemble.


            Optional Parameters
            -------------------    
            weights : ndarray, int
                A one dimensional array indicating the weights of the Boltzmann ensemble.
                If `None` (default), weight each structure equally.

            q_values : ndarray, float
                The values of |q| to compute the intensity profile for, in
                inverse Angstroms. Default: np.arange(0.02, 6.0, 0.02)

            Returns
            -------
            intensity_profile : ndarray, float
                An n x 2 array, where the first dimension is the magnitude |q| and
                the second is the average intensity at that point < I(|q|) >_phi.
            """

            # first, deal with weights
            if weights == None:
                weights = 1.0 # this will work later when we use array broadcasting
            else:
                if not len(weights) == trajectory.n_frames:
                    raise ValueError('length of `weights` array must be the same as the'
                                     'number of snapshots in `trajectory`')
                weights /= weights.sum()

            # next, construct the q-value array
            if q_values == None:
                q_values = np.arange(0.02, 6.0, 0.02)

            intensity_profile = np.zeros( ( len(q_values) , 2) )
            intensity_profile[:,0] = q_values

            # array that will hold the contribution from each snapshot in `trajectory`
            S = np.zeros(trajectory.n_frames)

            # extract the atomic numbers, number each atom by its type
            aZ = np.array([ a.element.atomic_number for a in trajectory.topology.atoms() ])
            n_atoms = len(aZ)

            atom_types = np.unique(aZ)
            num_atom_types = len(atom_types)
            cromermann = np.zeros(9*num_atom_types, dtype=np.float32)

            aid = np.zeros( n_atoms, dtype=np.int32 )
            atomic_formfactors = np.zeros( num_atom_types, dtype=np.float32 )

            for i,a in enumerate(atom_types):
                ind = i * 9
                try:
                    cromermann[ind:ind+9] = np.array(cromer_mann_params[(a,0)]).astype(np.float32)
                except KeyError as e:
                    logger.critical('Element number %d not in Cromer-Mann form factor parameter database' % a)
                    raise ValueError('Could not get critical parameters for computation')
                aid[ aZ == a ] = np.int32(i)

            # iterate over each value of q and compute the Debye scattering equation
            for q_ind,q in enumerate(q_values):
                print q_ind, len(q_values)

                # pre-compute the atomic form factors at this q
                qo = np.power( q / (4. * np.pi), 2)
                for ai in xrange(num_atom_types):                
                    for i in range(4):
                        atomic_formfactors[ai]  = cromermann[ai*9+8]
                        atomic_formfactors[ai] += cromermann[ai*9+i] * np.exp( cromermann[ai*9+i+5] * qo)

                # iterate over all pairs of atoms
                for i in range(n_atoms):
                    fi = atomic_formfactors[ aid[i] ]
                    for j in range(i+1, n_atoms):
                        fj = atomic_formfactors[ aid[j] ]

                        # iterate over all snapshots 
                        for k in range(trajectory.n_frames):
                            r_ij = np.linalg.norm(trajectory.xyz[k,i,:] - trajectory.xyz[k,j,:])
                            r_ij *= 10.0 # convert to angstroms.!
                            S[k] += 2.0 * fi * fj * np.sin( q * r_ij ) / ( q * r_ij )

                intensity_profile[q_ind,1] = np.sum( S * weights )

            return intensity_profile
            
        s = structure.load_coor(ref_file("3lyz.xyz"))
        qs = np.array([0.04, 2.0, 6.0])
        ref  = debye_reference(s, q_values=qs)
        calc = xray.debye(s, q_values=qs)
        assert_allclose(calc, ref)
    
    @skip    
    def test_against_scattering_simulation(self):
        if not GPU: raise SkipTest
        d = xray.Detector.generic()
        x = xray.Shot.simulate(self.t, 512, d)
        sim_ip = x.intensity_profile()
        debye = xray.debye(self.t, q_values=sim_ip[:,0])
        assert_allclose(debye, sim_ip)
        

if __name__ == '__main__':
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")
        test = TestShot()
        test.setup()
        test.test_pgc()
        test.test_pgr()
