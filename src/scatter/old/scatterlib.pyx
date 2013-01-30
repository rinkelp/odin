
"""
code shared by cpuscatter and gpuscatter cython wrappers
"""

import numpy as np
cimport numpy as np

from odin.refdata import cromer_mann_params

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
    
    
    
# cdef void extract_qxyz( np.ndarray[ndim=3, dtype=np.double_t] qxyz, 
#                         np.float_t * p_qx, np.float_t * p_qy, np.float_t * p_qz, 
#                         int nQ ):
#     cdef np.ndarray[ndim=1, dtype=np.float_t] qx = qxyz[:,0].astype(np.float32)
#     cdef np.ndarray[ndim=1, dtype=np.float_t] qy = qxyz[:,1].astype(np.float32)
#     cdef np.ndarray[ndim=1, dtype=np.float_t] qz = qxyz[:,2].astype(np.float32)
#     p_qx = &qx[0]
#     p_qy = &qy[0]
#     p_qz = &qz[0]
#     nQ = qxyz.shape[0]
#     return
# 
# 
# def extract_rxyz(rxyz):
#     cdef np.ndarray[ndim=1, dtype=np.float_t] rx = rxyz[:,0].flatten().astype(np.float32)
#     cdef np.ndarray[ndim=1, dtype=np.float_t] ry = rxyz[:,1].flatten().astype(np.float32)
#     cdef np.ndarray[ndim=1, dtype=np.float_t] rz = rxyz[:,2].flatten().astype(np.float32)
#     return rx, ry, rz
