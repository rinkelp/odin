
"""
Cython wrappers for GPU scattering.
"""

import numpy as np
cimport numpy as np

from scatterlib import *
from odin.refdata import cromer_mann_params


cdef extern from "gpuscatter.hh":
    cdef cppclass C_GPUScatter "GPUScatter":
        GPUScatter( int device_id_,
                    int bpg_,

                    int    nQ_,
                    float* h_qx_,
                    float* h_qy_,
                    float* h_qz_,

                    int    nAtoms_,
                    float* h_rx_,
                    float* h_ry_,
                    float* h_rz_,
                    int*   h_id_,

                    int    nCM_,
                    float* h_cm_,

                    int    nRot_,
                    float* h_rand1_,
                    float* h_rand2_,
                    float* h_rand3_,

                    int    nQout_,
                    float* h_outQ_
                    ) except +


def gpu_scatter(device_id, n_molecules, qxyz, rxyz, atomic_numbers, 
                poisson_parameter=0.0):
    """

    
    Parameters
    ----------
    device_id : int
    
    n_molecules : int
    
    qxyz : ndarray, float
    
    rxyz : ndarray, float
    
    atomic_numbers : ndarray, int
    
    poisson_parameter : float

    Returns
    -------
    self.intensities : ndarray, float
    """
    
    # stupidity check
    if type(device_id) != int:
        raise ValueError('device_id must be type int')
    
    # extract arrays from input
    nQ,     qx, qy, qx = extract_qxyz(qxyz)
    nAtoms, rx, ry, rx = extract_rxyz(rxyz)

    # see if we're going to use finite photon statistics
    if poisson_parameter == 0.0:
        finite_photons = 0
        pois = np.zeros(n_molecules, dtype=np.float32)
    else:
        finite_photons = 1
        pois = np.random.poisson(poisson_parameter, size=n_molecules)
    cdef np.ndarray[ndim=1, dtype=np.int_t] n_photons = pois.astype(np.float32)
    
    # make sure that we're working with a multiple of 512
    if not n_rotations % 512 == 0:
        raise ValueError('`n_rotations` must be a multiple of 512')
    bpg = 512 / n_rotations # blocks-per-grid
    
    cromermann, aid = get_cromermann_parameters(atomic_numbers)
    assert aid.shape[0] == rxyz.shape[0]
        
    cdef np.ndarray[ndim=1, dtype=np.float32] h_outQ = np.zeros(nQ, dtype=np.float32)
    rand1, rand2, rand3 = generate_random_floats(n_rotations)
        
    # call the actual CUDA code
    gpu_scatter_obj = new C_GPUScatter(int device_id, int bpg,
                int nQ, float* h_qx, float* h_qy, float* h_qz,
                int nAtoms, float* h_rx, float* h_ry, float* h_rz, int* h_id,
                int nCM, float* h_cm,
                int n_rotations, float* h_rand1, float* h_rand2, float* h_rand3,
                float* h_outQ)
    
    intensities = h_outQ.astype(np.float64) # convert back to double
    output_sanity_check(intensities)
            
    return intensities
