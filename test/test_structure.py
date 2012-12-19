
import os

from mdtraj import trajectory
from odin import structure
from odin.testing import ref_file

import numpy as np
from numpy.testing import assert_array_almost_equal

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
    n_samples = 50
    otraj = structure.multiply_conformations(traj, n_samples, 0.1)

    # iterate over x,y,z and check if any of the bins are more than 2.25 STD from the mean
    for i in [0,1,2]:
        h = np.histogram(otraj.xyz[:,0,i])[0]
        cutoff = h.std() * 2.25 # chosen arbitrarily
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

    
    
if __name__ == '__main__':
    m = test_m_confs()
    print m.n_frames
    m.save('test.pdb')
