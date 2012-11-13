
import gpuscatter
import numpy as np

from odin.data import cromer_mann_params
from odin.xray import Detector



def load_test_geometry():
    
    data = np.genfromtxt('../../example/goldBenchMark.coor')
    x = data[:,0].astype(np.float32)
    y = data[:,1].astype(np.float32)
    z = data[:,2].astype(np.float32)
    aid = data[:,3].astype(np.int32)
    
    return x, y, z, aid


def generate_rands(num_molecules):

    rand1 = np.random.rand(num_molecules).astype(np.float32)
    rand2 = np.random.rand(num_molecules).astype(np.float32)
    rand3 = np.random.rand(num_molecules).astype(np.float32)

    return rand1, rand2, rand3


def scatter():
    
    num_molecules = 512
    
    # choose the number of molecules (must be multiple of 512)
    num_molecules = num_molecules - (num_molecules % 512)
    bpg = num_molecules / 512
    print "new nmolec, bpg", num_molecules, bpg
    
    # get detector
    detector = Detector.generic_detector()
    qx = detector.reciprocal[:,0].astype(np.float32)
    qy = detector.reciprocal[:,1].astype(np.float32)
    qz = detector.reciprocal[:,2].astype(np.float32)
    num_q = len(qx)
    
    # get atomic positions
    rx, ry, rz, aid = load_test_geometry()
    num_atoms = len(rx)
    atom_types = np.unique(aid)
    num_atom_types = len(atom_types)
    
    # get cromer-mann parameters for each atom type
    cromermann = np.zeros(9*num_atom_types, dtype=np.float32)
    for i,a in enumerate(atom_types):
        ind = i * 9
        cromermann[ind:ind+9] = cromer_mann_params[(a,0)]
        aid[ aid == a ] = i # make the atom index 0, 1, 2, ...

    print aid
    print cromermann
 
    # get random numbers
    rand1, rand2, rand3 = generate_rands(num_molecules)
    
    # run dat shit
    out_obj = gpuscatter.GPUScatter(bpg, qx, qy, qz,
                                    rx, ry, rz, aid,
                                    cromermann,
                                    rand1, rand2, rand3, num_q)
    
    output = out_obj.this[1].astype(np.float64)

    return output



if __name__ == '__main__':
    print scatter()

