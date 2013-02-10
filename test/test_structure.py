
import os

from mdtraj import trajectory
from odin import structure
from odin.testing import ref_file

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()

def test_m_confs():
    t = trajectory.load( ref_file('ala2.pdb') )
    m = structure.multiply_conformations(t, 10, 1.0)
    return m
    
def test_rm_com():
    t = trajectory.load( ref_file('ala2.pdb') )
    r = structure.remove_COM(t)
    
    masses = [ a.element.mass for a in t.topology.atoms() ]
    
    for i in range(t.n_frames):
        assert_array_almost_equal(np.zeros(3), np.average(t.xyz[i,:,:], weights=masses, axis=0))
        
def test_multiply_conformations():
    traj = structure.load_coor(ref_file('goldBenchMark.coor'))
    n_samples = 150
    otraj = structure.multiply_conformations(traj, n_samples, 0.1)

    # iterate over x,y,z and check if any of the bins are more than 3 STD from the mean
    for i in [0,1,2]:
        h = np.histogram(otraj.xyz[:,0,i])[0]
        cutoff = h.std() * 3.0 # chosen arbitrarily
        deviations = np.abs(h - h.mean())
        print deviations / h.std()
        if np.any( deviations > cutoff ):
            raise RuntimeError('Highly unlikely centers of mass are randomly '
                               'distributed in space. Test is stochastic, though, so'
                               ' try running again to make sure you didn\'t hit a '
                               'statistical anomaly')

        # use the below to visualize the result
        # plt.hist(otraj.xyz[:,0,i])
        # plt.show()
        
def test_load_coor():
    
    s = structure.load_coor( ref_file('goldBenchMark.coor') )
    s.save('s.pdb')
    t = trajectory.load('s.pdb')
    
    assert_array_almost_equal(s.xyz, t.xyz, decimal=3)
    
    for a in t.topology.atoms():
        assert a.element.symbol == 'Au'
        
    if os.path.exists('s.pdb'):
        os.remove('s.pdb')
        
        
def test_random_rotation():
    
    # generate a bunch of random vectors that are distributed over the unit
    # sphere and test that the distributions of their angles is uniform
    
    x_unit = np.zeros(3)
    x_unit[0] = 1.0
    
    n_samples = int(1e4)
    n_bins = 10
    
    phi_x = np.zeros(n_samples)
    phi_y = np.zeros(n_samples)
    
    for i in range(n_samples):
        v = structure.quaternion.rand_rotate_vector(x_unit)
        phi_y[i] = np.arctan2(v[1], v[0])
        phi_x[i] = np.arctan2(v[2], v[0])
        
    vx, b = np.histogram(phi_x, bins=n_bins)
    vy, b = np.histogram(phi_y, bins=n_bins)
    
    v_ref = np.ones(n_bins) * float(n_samples / n_bins)
    
    assert_allclose(vx, v_ref, rtol=1e-1)
    assert_allclose(vy, v_ref, rtol=1e-1)
    