
"""
Cython wrappers for CPU scattering.
"""

import numpy as np
cimport numpy as np

from odin.refdata import cromer_mann_params
from odin import installed

def get_cromermann_parameters(atomic_numbers, max_num_atom_types=None):
    """
    Get cromer-mann parameters for each atom type and renumber the atom 
    types to 0, 1, 2, ... to point to their CM params.
    
    Parameters
    ----------
    atomic_numbers : ndarray, int
        A numpy array of the atomic numbers of each atom in the system.
        
    max_num_atom_types : int
        The maximium number of atom types allowable
    
    Returns
    -------
    cromermann : c-array, float
        The Cromer-Mann parameters for the system. Positions [(0-8) * aid] are
        reserved for each atom type in the system (see `aid` below).
        
    aid : c-array, int
        The indicies of the atomic id's of each atom in the system. This is an
        arbitrary compressed index for use within the scattering code. Really
        this is just a renumbering so that each atom type recieves an index
        0, 1, 2, ... corresponding to the position of that atom's type in
        the `cromermann` array.  
    """

    atom_types = np.unique(atomic_numbers)
    num_atom_types = len(atom_types)

    if num_atom_types:
        if num_atom_types > max_num_atom_types:
            raise Exception('Fatal Error. Your molecule has too many unique atom  '
                            'types -- the scattering code cannot handle more due'
                            ' to code requirements. You can recompile the kernel'
                            ' to fix this -- see file odin/src/scatter. Email'
                            'tjlane@stanford.edu complaining about shitty code.')

    cromermann = np.zeros( 9*num_atom_types, dtype=np.float32 )
    aid = np.zeros( len(atomic_numbers), dtype=np.int32 )

    for i,a in enumerate(atom_types):
        ind = i * 9
        try:
            cromermann[ind:ind+9] = np.array(cromer_mann_params[(a,0)]).astype(np.float32)
        except KeyError as e:
            print 'Element number %d not in Cromer-Mann form factor parameter database' % a
            raise ValueError('Could not get critical parameters for computation')
        aid[ atomic_numbers == a ] = np.int32(i)

    nCM = len(cromermann)

    return cromermann, aid


def output_sanity_check(intensities):
    """
    Perform a basic sanity check on the intensity array we're about to return.
    """
    
    # check for NaNs in output
    if np.isnan(np.sum(intensities)):
        raise RuntimeError('Fatal error, NaNs detected in scattering output!')
        
    # check for negative values in output
    if len(intensities[intensities < 0.0]) != 0:
        raise RuntimeError('Fatal error, negative intensities detected in scattering output!')
        
    return
    

cdef extern from "gpuscatter.hh":
    cdef cppclass C_GPUScatter "GPUScatter":
        C_GPUScatter(int    device_id_,
                     int    bpg_,
                     int    nQ_,
                     float* h_qx_,
                     float* h_qy_,
                     float* h_qz_,
                     int    nAtoms_,
                     float* h_rx_,
                     float* h_ry_,
                     float* h_rz_,
                     np.int32_t*   h_id_,
                     int    nCM_,
                     float* h_cm_,
                     int    nRot_,
                     float* h_rand1_,
                     float* h_rand2_,
                     float* h_rand3_,
                     float* h_outQ_ ) except +
                    
cdef extern from "cpuscatter.hh":
    cdef cppclass C_CPUScatter "CPUScatter":
        C_CPUScatter(int    nQ_,
                     float* h_qx_,
                     float* h_qy_,
                     float* h_qz_,
                     int    nAtoms_,
                     float* h_rx_,
                     float* h_ry_,
                     float* h_rz_,
                     np.int32_t*   h_id_,
                     int    nCM_,
                     float* h_cm_,
                     int    nRot_,
                     float* h_rand1_,
                     float* h_rand2_,
                     float* h_rand3_,
                     float* h_outQ_ ) except +

                     
cdef C_CPUScatter * cpu_scatter_obj
cdef C_GPUScatter * gpu_scatter_obj                     
                     
                     
def scatter(n_molecules, qxyz, rxyz, atomic_numbers, poisson_parameter=0.0,
            device=0):
    """
    Parameters
    ----------
    n_molecules : int
    
    qxyz : ndarray, float
    
    rxyz : ndarray, float
    
    atomic_numbers : ndarray, int
    
    poisson_parameter : float
    
    device : int OR "CPU"
        

    Returns
    -------
    self.intensities : ndarray, float
        A flat array of the simulated intensities, each position corresponding
        to a scattering vector from `qxyz`.
    """
    
    # check to see if we're going to run on the GPU or CPU
    if type(device) == int: # we're running on the GPU
        if not installed.gpuscatter:
            raise ImportError('gpuscatter.cu was not installed, cannot run on GPU')            
        if not n_molecules % 512 == 0:
            raise ValueError('`n_rotations` must be a multiple of 512')
        bpg = 512 / n_molecules # blocks-per-grid
        GPU = True
    elif device.lower() == 'cpu':  # we're running on the CPU
        GPU = False
    else:
        raise ValueError('device must be int (GPU device id) or "CPU"')
    
    # extract arrays from input
    cdef np.ndarray[ndim=1, dtype=np.float32_t] rx = rxyz[:,0].astype(np.float32)
    cdef np.ndarray[ndim=1, dtype=np.float32_t] ry = rxyz[:,1].astype(np.float32)
    cdef np.ndarray[ndim=1, dtype=np.float32_t] rz = rxyz[:,2].astype(np.float32)
    
    cdef np.ndarray[ndim=1, dtype=np.float32_t] qx = qxyz[:,0].astype(np.float32)
    cdef np.ndarray[ndim=1, dtype=np.float32_t] qy = qxyz[:,1].astype(np.float32)
    cdef np.ndarray[ndim=1, dtype=np.float32_t] qz = qxyz[:,2].astype(np.float32)
    
    # generate random numbers
    cdef np.ndarray[ndim=1, dtype=np.float32_t] rand1 = np.random.rand(n_molecules).astype(np.float32)
    cdef np.ndarray[ndim=1, dtype=np.float32_t] rand2 = np.random.rand(n_molecules).astype(np.float32)
    cdef np.ndarray[ndim=1, dtype=np.float32_t] rand3 = np.random.rand(n_molecules).astype(np.float32)


    # see if we're going to use finite photon statistics
    if poisson_parameter == 0.0:
        finite_photons = 0
        pois = np.zeros(n_molecules, dtype=np.float32)
    else:
        finite_photons = 1
        pois = np.random.poisson(poisson_parameter, size=n_molecules)
    cdef np.ndarray[ndim=1, dtype=np.int_t] n_photons = pois.astype(np.float32)


    # get the Cromer-Mann parameters
    py_cromermann, py_aid = get_cromermann_parameters(atomic_numbers)
    cdef np.ndarray[ndim=1, dtype=np.float32_t] cromermann = py_cromermann.astype(np.float32)
    cdef np.ndarray[ndim=1, dtype=np.int32_t] aid = py_aid
    assert aid.shape[0] == rxyz.shape[0]

    
    # initialize output array
    cdef np.ndarray[ndim=1, dtype=np.float32_t] h_outQ = np.zeros(qxyz.shape[0], dtype=np.float32)

        
    # call the actual C++ code
    if GPU:
        gpu_scatter_obj = new C_GPUScatter(device, bpg,
                   qxyz.shape[0], &qx[0], &qy[0], &qz[0],
                   rxyz.shape[0], &rx[0], &ry[0], &rz[0], &aid[0],
                   len(cromermann), &cromermann[0],
                   n_molecules, &rand1[0], &rand2[0], &rand3[0],
                   &h_outQ[0])
        del gpu_scatter_obj
    else:
        cpu_scatter_obj = new C_CPUScatter(qxyz.shape[0], &qx[0], &qy[0], &qz[0],
                                   rxyz.shape[0], &rx[0], &ry[0], &rz[0], &aid[0],
                                   len(cromermann), &cromermann[0],
                                   n_molecules, &rand1[0], &rand2[0], &rand3[0],
                                   &h_outQ[0])
        del cpu_scatter_obj
                                   
    # deal with the output
    output_sanity_check(h_outQ)
    
    return

