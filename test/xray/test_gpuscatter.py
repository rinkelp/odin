
import gpuscatter
import numpy as np

from odin.data import cromer_mann_params
from odin.xray import Detector



def load_test_geometry():
    
    data = np.genfromtxt('../../example/goldBenchMark.coor')
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    aid = data[:,3]
    
    return x, y, z, aid


def generate_rands(num_molecules):

    rand1 = np.random.rand(num_molecules)
    rand2 = np.random.rand(num_molecules)
    rand3 = np.random.rand(num_molecules)

    return rand1, rand2, rand3


def scatter():
    
    num_molecules = 23493
    
    # choose the number of molecules (must be multiple of 512)
    num_molecules = num_molecules - (num_molecules % 512)
    bpg = num_molecules / 512
    print "new nmolec, bpg", num_molecules, bpg
    
    # get detector
    detector = Detector.generic_detector()
    qx = detector.reciprocal[:,0]
    qy = detector.reciprocal[:,1]
    qz = detector.reciprocal[:,2]
    num_q = len(qx)
    
    # get atomic positions
    rx, ry, rz, aid = load_test_geometry()
    num_atoms = len(rx)
    atom_types = np.unique(aid)
    num_atom_types = len(atom_types)
    
    # get cromer-mann parameters for each atom type
    cromermann = np.zeros(9*num_atom_types)
    for i,a in enumerate(atom_types):
        ind = i * 9
        cromermann[ind:ind+9] = cromer_mann_params[(a,0)]
    
    # get random numbers
    rand1, rand2, rand3 = generate_rands(num_molecules)
    
    # initialize output array
    output = np.zeros(num_q)
    
    # run dat shit
    sobj = gpuscatter.GPUScatter(bpg, num_q, qx, qy, qz,
                                  num_atoms, num_atom_types, rx, ry, rz, aid,
                                  cromermann, rand1, rand2, rand3,
                                  output)
    sobj.run()
    sobj.retreive()
    
    print output
    
    return



if __name__ == '__main__':
    scatter()

