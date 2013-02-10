
"""
Library for performing simulations of x-ray scattering experiments.
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

import numpy as np
from threading import Thread

from odin import cpuscatter
from odin.refdata import cromer_mann_params

try:
    from odin import gpuscatter
    GPU = True
except ImportError as e:
    GPU = False


def simulate_shot(traj, num_molecules, detector, traj_weights=None,
                  force_no_gpu=False, device_id=0):
    """
    Simulate a scattering 'shot', i.e. one exposure of x-rays to a sample.
    
    Assumes we have a Boltzmann distribution of `num_molecules` identical  
    molecules (`trajectory`), exposed to a beam defined by `beam` and projected
    onto `detector`.
    
    Each conformation is randomly rotated before the scattering simulation is
    performed. Atomic form factors from X, finite photon statistics, and the 
    dilute-sample (no scattering interference from adjacent molecules) 
    approximation are employed.
    
    Parameters
    ----------
    traj : mdtraj.trajectory
        A trajectory object that contains a set of structures, representing
        the Boltzmann ensemble of the sample. If len(traj) == 1, then we assume
        the sample consists of a single homogenous structure, replecated 
        `num_molecules` times.
        
    detector : odin.xray.Detector OR ndarray, float
        A detector object the shot will be projected onto. Can alternatively be
        just an n x 3 array of vectors to project onto.
        
    num_molecules : int
        The number of molecules estimated to be in the `beam`'s focus.
        
    Optional Parameters
    -------------------
    traj_weights : ndarray, float
        If `traj` contains many structures, an array that provides the Boltzmann
        weight of each structure. Default: if traj_weights == None, weights
        each structure equally.
        
    force_no_gpu : bool
        Run the (slow) CPU version of this function.
        
    device_id : int
        The index of the GPU device to run on.
        
    Returns
    -------
    intensities : ndarray, float
        An array of the intensities at each pixel of the detector.
        
    See Also
    --------
    odin.xray.Shot.simulate()
    odin.xray.Shotset.simulate()
        These are factory functions that call this function, and wrap the
        results into the Shot and Shotset classes, respectively.
    """
        
    logger.debug('Performing scattering simulation...')
    logger.debug('Simulating %d copies in the dilute limit' % num_molecules)

    if traj_weights == None:
        traj_weights = np.ones( traj.n_frames )
    traj_weights /= traj_weights.sum()
        
    num_per_shapshot = np.random.multinomial(num_molecules, traj_weights)
        
    # get the scattering vectors
    if isinstance(detector, odin.xray.Detector):    
        qxyz = detector.reciprocal
        assert detector.num_pixels == qxyz.shape[0]
    elif isinstance(detector, np.ndarray):
        qxyz = detector
    else:
        raise ValueError('`detector` must be {odin.xray.Detector, np.ndarray}')
    num_q = qxyz.shape[0]
    
    # extract the atomic numbers
    atomic_numbers = np.array([ a.element.atomic_number for a in traj.topology.atoms() ])
        
    # iterate over snapshots in the trajectory
    intensities = np.zeros(num_q)
    for i,num in enumerate(num_per_shapshot):
        num = int(num)
        if num > 0: # else, we can just skip...
        
            # pull xyz coords
            rxyz = traj.xyz[i,:,:] * 10.0 # convert nm -> ang.

            # choose the number of molecules & divide work between CPU & GPU
            # GPU is fast but can only do multiples of 512 molecules - run
            # the remainder on the CPU
            if force_no_gpu or (not GPU):
                num_cpu = num
                num_gpu = 0
                bpg = 0
                logger.warning('Forced "no GPU": running CPU-only computation')
            else:
                num_cpu = num % 512
                num_gpu = num - num_cpu
            
            logger.info('Running %d molecules from snapshot %d...' % (num, i))  

            # multiprocessing cannot return values, so generate a helper function
            # that will dump returned values into a shared array
            threads = []
            def multi_helper(compute_device, fargs):
                """ a helper function that performs either CPU or GPU calcs """
                if compute_device == 'cpu':
                    from odin import cpuscatter
                    func = cpuscatter.simulate
                elif compute_device == 'gpu':
                    from odin import gpuscatter
                    func = gpuscatter.simulate
                else:
                    raise ValueError('`compute_device` should be one of {"cpu",\
                     "gpu"}, was: %s' % compute_device)
                intensities[:] += func(*fargs)
                return

            # run dat shit
            if num_cpu > 0:
                logger.debug('Running CPU scattering code (%d/%d)...' % (num_cpu, num))
                cpu_args = (num_cpu, qxyz, rxyz, atomic_numbers)
                t_cpu = Thread(target=multi_helper, args=('cpu', cpu_args))
                t_cpu.start()
                threads.append(t_cpu)                

            if num_gpu > 0:
                logger.debug('Sending calculation to GPU device...')
                gpu_args = (num_gpu, qxyz, rxyz, atomic_numbers, device_id)
                t_gpu = Thread(target=multi_helper, args=('gpu', gpu_args))
                t_gpu.start()
                threads.append(t_gpu)
                
            # ensure child processes have finished
            for t in threads:
                t.join()
        
    return intensities
        
        
def atomic_formfactor(atomic_Z, q_mag):
    """
    Compute the (real part of the) atomic form factor.
    
    Parameters
    ----------
    atomic_Z : int
        The atomic number of the atom to compute the form factor for.
        
    q_mag : float
        The magnitude of the q-vector at which to evaluate the form factor.
        
    Returns
    -------
    fi : float
        The real part of the atomic form factor.
    """
        
    qo = np.power( q_mag / (4. * np.pi), 2)
    cromermann = cromer_mann_params[(atomic_Z,0)]
        
    fi = cromermann[8]
    for i in range(4):
        fi += cromermann[i] * np.exp( - cromermann[i+4] * qo)
        
    return fi


def debye(trajectory, weights=None, q_values=None):
    """
    Computes the Debye scattering equation for the structures in `trajectory`,
    producing the theoretical intensity profile.

    Treats the object `trajectory` as a sample from a Boltzmann ensemble, and
    averages the profile from each snapshot in the trajectory. If `weights` is
    provided, weights the ensemble accordingly -- otherwise, all snapshots are
    given equal weights

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
        weights = np.ones(trajectory.n_frames)
    else:
        if not len(weights) == trajectory.n_frames:
            raise ValueError('length of `weights` array must be the same as the'
                             'number of snapshots in `trajectory`')
        weights /= weights.sum()

    # next, construct the q-value array
    if q_values == None:
        q_values = np.arange(0.02, 6.0, 0.02)

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

    # construct pointers for weave code
    xyz = trajectory.xyz.flatten().astype(np.float32) * 10.0 # convert to angstroms.!
    n_frames = trajectory.n_frames
    intensities = np.zeros(len(q_values), dtype=np.float32)
    n_q_values = len(q_values)
    q_values = q_values.astype(np.float32)
    weights = weights.astype(np.float32)
        
    weave.inline(r"""
    Py_BEGIN_ALLOW_THREADS
    
    // iterate over each value of q and compute the Debye scattering equation
    
    // #pragma omp parallel for shared(intensities, atomic_formfactors, aid, weights, xyz, q_values, n_atoms, n_frames)
    for (int qi = 0; qi < n_q_values; qi++) {
    
        // pre-compute the atomic form factors at this q
        float q = q_values[qi];
        float qo = q*q / (16.0 * M_PI * M_PI);

        for (int ai = 0; ai < num_atom_types; ai++) {

            int tind = ai * 9;
            float f;

            f =  cromermann[tind]   * exp(-cromermann[tind+4]*qo);
            f += cromermann[tind+1] * exp(-cromermann[tind+5]*qo);
            f += cromermann[tind+2] * exp(-cromermann[tind+6]*qo);
            f += cromermann[tind+3] * exp(-cromermann[tind+7]*qo);
            f += cromermann[tind+8];

            atomic_formfactors[ai] = f;
        }
            
        // iterate over all pairs of atoms
        for (int i = 0; i < n_atoms; i++) {
            float fi = atomic_formfactors[ aid[i] ];
            for (int j = i+1; j < n_atoms; j++) {
                float fj = atomic_formfactors[ aid[j] ];
                
                // iterate over all snapshots 
                for (int k = 0; k < n_frames; k++) {
                
                    // compute the atom-atom distance
                    float r_ij, S, d;
                    
                    r_ij = 0.0;
                    for (int l = 0; l < 3; l++) {
                        int ind_ri = k*n_atoms + i*3 + l;
                        int ind_rj = k*n_atoms + j*3 + l;
                        d = xyz[ind_ri] - xyz[ind_rj];
                        r_ij += d*d;
                    }
                    r_ij = sqrt(r_ij);
                    
                    S = 2.0 * fi * fj * sin( q * r_ij ) / ( q * r_ij );
                    
                    // add the result to our final output array
                    #pragma omp critical(intensities_update)
                    intensities[qi] += S * weights[k];
                }
            }            
        }
    }
    Py_END_ALLOW_THREADS
    """, ['q_values', 'cromermann', 'aid', 'xyz', 'intensities', 
          'atomic_formfactors', 'weights', 'n_q_values', 'num_atom_types', 
          'n_atoms', 'n_frames'],
          extra_link_args = ['-lgomp'],
          extra_compile_args = ["-O3", "-fopenmp"] )

    # tjl, later remove this and just stack the results
    intensity_profile = np.zeros( ( len(q_values) , 2) )
    intensity_profile[:,0] = q_values
    intensity_profile[:,1] = intensities

    return intensity_profile


def sph_hrm_coefficients(trajectory, weights=None, q_magnitudes=None, 
                         num_coefficients=10):
    """
    Numerically evaluates the coefficients of the projection of a structure's
    fourier transform onto the three-dimensional spherical harmonic basis. Can
    be used to directly compare a proposed structure to the same coefficients
    computed from correlations in experimentally observed scattering profiles.
    
    Parameters
    ----------
    trajectory : mdtraj.trajectory
        A trajectory object representing a Boltzmann ensemble.
        
    weights : ndarray, float
        A list of weights, for how to weight each snapshot in the trajectory.
        If not provided, treats each snapshot with equal weight.
        
    q_magnitudes : ndarray, float
        A list of the reciprocal space magnitudes at which to evaluate 
        coefficients.
        
    num_coefficients : int
        The order at which to truncate the spherical harmonic expansion
    
    Returns
    -------
    sph_coefficients : ndarray, float
        A 3-dimensional array of coefficients. The first dimension indexes the
        order of the spherical harmonic. The second two dimensions index the
        array `q_magnitudes`.
    """
    
    logger.info('Projecting image into spherical harmonic basis...')
    
    # first, deal with weights
    if weights == None:
        weights = np.ones(trajectory.n_frames)
    else:
        if not len(weights) == trajectory.n_frames:
            raise ValueError('length of `weights` array must be the same as the'
                             'number of snapshots in `trajectory`')
        weights /= weights.sum()
    
    # initialize the q_magnitudes array
    if q_magnitudes == None:
        q_magnitudes = np.arange(1.0, 6.0, 0.02)
    num_q_mags = len(q_magnitudes)
    
    # don't do odd values of ell
    l_vals = range(0, 2*num_coefficients, 2)
    
    # initialize output space
    sph_coefficients = np.zeros((num_coefficients, num_q_mags, num_q_mags))
    Slm = np.zeros(( num_coefficients, 2*num_coefficients+1, num_q_mags), 
                     dtype=np.complex128 )
    
    # get the quadrature vectors we'll use, a 900 x 4 array : [q_x, q_y, q_z, w]
    from odin.refdata import sph_quad_900
    q_phi   = arctan3(sph_quad_900[:,1], sph_quad_900[:,0])
        
    # iterate over all snapshots in the trajectory
    for i in range(trajectory.n_frames):
        
        for iq,q in enumerate(q_magnitudes):
            logger.info('Computing coefficients for q=%f\t(%d/%d)' % (q, iq+1, num_q_mags))
                    
            # compute S, the single molecule scattering intensity
            S = simulate_shot(trajectory, 1, sph_quad_900[:,:3] * q, 
                              force_no_gpu=True)

            # project S onto the spherical harmonics using spherical quadrature
            for il,l in enumerate(l_vals):                
                for m in range(-l, l+1):
                    
                    N = np.sqrt( 2. * l * factorial(l-m) / ( 4. * np.pi * factorial(l+m) ) )
                    Plm = special.lpmv(m, l, sph_quad_900[:,2])
                    Ylm = N * np.exp( 1j * m * q_phi ) * Plm
                    
                    Slm[il, m, iq] = np.sum( S * Ylm * sph_quad_900[:,3] )
    
        # now, reduce the Slm solution to C_l(q1, q2)
        for iq1, q1 in enumerate(q_magnitudes):
            for iq2, q2 in enumerate(q_magnitudes):
                for il, l in enumerate(l_vals):
                    sph_coefficients[il, iq1, iq2] += weights[i] * \
                                                      np.real( np.sum( Slm[il,:,iq1] *\
                                                      np.conjugate(Slm[il,:,iq2]) ) )
    
    return sph_coefficients
