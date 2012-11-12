# THIS FILE IS PART OF ODIN


"""
Classes, methods, functions for use with xray scattering experiments.

Todo:
-- add simulate_shot factor functions to Shot and Shotset

"""

import numpy as np
from odin.data import cromer_mann_params
from bisect import bisect_left

from odin import utils

import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# FUNDAMENTAL CONSTANTS

h = 4.135677516e-15   # Planks constant | eV s
c = 299792458         # speed of light  | m / s

# ------------------------------------------------------------------------------


class Beam(object):
    """
    Class that converts energies, wavelengths, frequencies, and wavenumbers.
    Each instance of this class can represent a light source.
    
    Attributes
    ----------
    self.energy      (keV)
    self.wavelength  (angstroms)
    self.frequency   (Hz)
    self.wavenumber  (angular, inv. angstroms)
    """
    
    def __init__(self, flux, **kwargs):
        """
        Generate an instance of the Beam class.
        
        Parameters
        ----------
        flux : float
            The photon flux in the focal point of the beam.
        
        **kwargs : dict
            Exactly one of the following, in the indicated units
            -- energy:     keV
            -- wavelength: angstroms
            -- frequency:  Hz
            -- wavenumber: inverse angstroms
        """
        
        self.flux = flux
        
        # make sure we have only one argument
        if len(kwargs) != 1:
            raise ValueError('Expected exactly one argument, got %d' % (len(args)+1) )
        
        self.units = 'energy: keV, wavelengths: angstroms, frequencies: Hz, wavenumbers: inverse angstroms'
        
        # no matter what gets provided, go straight to energy and then
        # convert to the rest from there
        for key in kwargs:
                            
            if key == 'energy':
                self.energy = float(kwargs[key])
            
            elif key == 'wavenumber':
                self.wavenumber = float(kwargs[key])
                self.energy = self.wavenumber * h * c * 10.**10. / (2.0 * np.pi)
                
            elif key == 'wavelength':
                self.wavelength = float(kwargs[key])
                self.energy = h * c * 10.**10. / self.wavelength
                
            elif key == 'frequency':
                self.frequency = float(kwargs[key])
                self.energy = self.frequency * h / 1000.
                
            else:
                raise ValueError('%s not a recognized kwarg' % key)
        
        # perform the rest of the conversions
        self.wavelength = h * c * 10.**10. / self.energy
        self.wavenumber = 2.0 * np.pi / self.wavelength
        self.frequency = self.energy * (1000. / h)
        
        # some aliases
        self.k = self.wavenumber
    
    
class Detector(Beam):
    """
    Class that provides a plethora of geometric specifications for a detector
    setup. Also provides loading and saving of detector geometries.
    """
    
    def __init__(self, xyz_file, path_length, *args):
        """
        Instantiate a Detector object.
        
        Detector objects provide a handle for the many representations of
        detector geometry in scattering experiments, namely:
        
        -- real space
        -- real space in polar coordinates
        -- reciprocal space (q-space)
        -- reciprocal space in polar coordinates (q, theta, phi)
        
        Parameters
        ----------
        xyz_file : str
            A string pointing to a file containing the x,y,z coordinates of each
            pixel of the detector. File should be in X format.
            
        path_length : float
            The distance between the sample and the detector, in the same units
            as the pixel dimensions in the input file.
            
        k : float
            The wavenumber of the incident beam to use.
            
        Optional Parameters
        -------------------
        photon : odin.xray.Beam
            A Beam object, defining the beam energy. If this is passed, the
            the argument `k` for the wavenumber is redundant and should not be
            passed.
        """
        
        self.file_path = xyz_file
        self.path_length = path_length
        
        # parse the wavenumber
        if len(args) != 1:
            raise ValueError('Expected exactly two arguments, got %d' % (len(args)+1) )
        else:
            for arg in args:
                if isinstance(arg, Beam):
                    self.k = arg.wavenumber
                elif type(arg) == float:
                    self.k = arg
                else:
                    raise ValueError('Must pass `beam` or `k` argument')
                    
        # load the real space representation, origin at the middle of the
        # detector (presumably)
        self.xyz = self.loadf(self.file_path)
        
        # convert to other reps
        self.real = self.xyz
        self.real[:,2] += self.path_length # set origin to sample
        
        self.polar      = self.real_to_polar(self.xyz)
        self.reciprocal = self.real_to_reciprocal(self.xyz)
        self.recpolar   = self.real_to_recpolar(self.xyz)
        
        
    def loadf(self, filename):
        """
        Loads a file specifying a detector geometry and returns an array of the
        xyz corrdinates of each of the pixel positions on the detector.
        
        Parameters
        ----------
        filename : str
            The filename of file containing the detector geometry in X format.
            
        Returns
        -------
        xyz : ndarray, float
            An n x 3 array, specifing the x,y,z positions of n detector pixels.
        """
        
        try:
            xyz = np.genfromtxt(filename)
        except:
            logger.error('Could not find method capable of loading %s' % filename)
        
        assert xyz.shape[1] == 3 # 3 coordinates for each pixel
        
        return xyz
        
            
    def real_to_polar(self, xyz):
        """
        Convert the real-space representation to polar coordinates.
        """
        xyz[:,2] += self.path_length # set origin to sample
        polar = self._to_polar(xyz)
        
        return polar
        
            
    def real_to_reciprocal(self, xyz):
        """
        Convert the real-space to reciprocal-space in cartesian form.
        """
        
        # generate unit vectors in the pixel direction, origin at sample
        S = self.xyz.copy()
        S[:,2] += self.path_length # set origin to sample
        S = self._unit_vector(S)
        
        # generate unit vectors in the z-direction
        S0 = np.zeros(xyz.shape)
        S0[:,2] = np.ones(xyz.shape[0])
        
        q = self.k * (S - S0)
        
        return q
        
        
    def real_to_recpolar(self, xyz):
        """
        Convert the real-space to reciprocal-space in polar form, that is
        (|q|, theta, phi).
        """
        
        q = self.real_to_reciprocal(xyz)
        reciprocal_polar = self._to_polar(q)
        
        return reciprocal_polar
        
        
    def _norm(self, vector):
        """
        Compute the norm of an n x m array of vectors, where m is the dimension.
        """
        return np.sqrt( np.sum( np.power(vector, 2), axis=1 ) )
        
        
    def _unit_vector(self, vector):
        """
        Returns a unit-norm version of `vector`.
        
        Parameters
        ----------
        vector : ndarray, float
            An n x m vector of floats, where m is assumed to be the dimension
            of the space.
            
        Returns
        -------
        unit_vectors : ndarray,float
            An n x m vector, same as before, but now of unit length
        """

        norm = self._norm(vector)
        
        unit_vectors = np.zeros( vector.shape )
        for i in range(vector.shape[1]):
            unit_vectors[:,i] = vector[:,i] / norm
        
        return unit_vectors


    def _to_polar(self, vector):
        """
        Converts an n m-dimensional `vector`s to polar coordinates
        """
        
        polar = np.zeros( vector.shape )
        
        polar[:,0] = self._norm(vector)
        polar[:,1] = np.arccos(vector[:,2] / polar[:,0]) # cos^{-1}(z/r)
        
        # for the phi angle, use arctan2. If the value is in QII or QIII,
        # this returns a negative angle in [0,pi]. We probably want one in
        # [pi,2pi] so we adjust accordingly
        polar[:,2] = np.arctan2(vector[:,1], vector[:,0]) # y coord first!
        
        neg_ind = np.where( polar[:,2] < 0 )
        polar[neg_ind,2] += 2 * np.pi
        
        return polar
        
    @classmethod
    def generic_detector(cls, spacing=0.02, lim=1.0, k=4.33, flux=100.0, l=1.0):
        """
        Generates a simple grid detector that can be used for testing.
        (Factory function.)

        Optional Parameters
        -------------------
        spacing : float
            The real-space grid spacing
        lim : float
            The upper and lower limits of the grid
        k : float
            Wavenumber of the beam
        l : float
            The path length from the sample to the detector

        Returns
        -------
        detector : odin.xray.Detector
            An instance of the detector that meets the specifications of the 
            parameters
        """

        beam = Beam(flux, wavenumber=k)

        x = np.arange(-lim, lim, spacing)
        xx, yy = np.meshgrid(x, x)

        xyz = np.zeros((len(x)**2, 3))
        xyz[:,0] = xx.flatten()
        xyz[:,1] = yy.flatten()

        np.savetxt('test_detector.dat', xyz)

        detector = Detector('test_detector.dat', l, beam)

        return detector
        
        
class Shot(object):
    """
    Objects representing the data contained in a single xray exposure, and
    associated methods for analyzing such data.
    
    See Also
    --------
    Shotset : odin.xray.Shotset
        A collection of shots, with methods for analyzing statistical properties
        across the collection
    """
    
    def __init__(self, intensities, detector):
        """
        Instantiate a Shot class.
        
        Parameters
        ----------
        intensities : ndarray, float
            A one-dimensional array of the measured intensities at each pixel
            of the detector.
        
        detector : odin.xray.Detector
            A detector object, containing the pixel positions in space.
        """
        
        self.intensities = self._interpolate_to_polar(intensities, detector)
        self._mask_missing()
        
        
    def _interpolate_to_polar(self, intensities, detector, 
                              q_spacing=0.02, phi_spacing=1.0):
        """
        Interpolate our (presumably) cartesian-based measurements into a binned
        polar coordiante system.
        
        Parameters
        ----------
        intensities : ndarray, float
            A one-dimensional array of the measured intensities at each pixel
            of the detector.
        
        detector : odin.xray.Detector
            A detector object, containing the pixel positions in space.
        
        Optional Parameters
        -------------------
        q_spacing : float
            The q-vector spacing, in inverse angstroms.
        
        phi_spacing : float
            The azimuthal angle spacing, IN DEGREES.
        
        Returns
        -------
        interpoldata : ndarray, float
            An n x 3 array, where n in the total number of data points.
                interpoldata[:,0] -- magnitude of q, |q|
                interpoldata[:,1] -- azmuthal angle phi
                interpoldata[:,2] -- intensity I(|q|, phi)
        """
        
        # determine the bounds of the grid, and the discrete points to use
        self.phi_spacing = phi_spacing
        self.num_phi = int( 360. / self.phi_spacing )
        self.phi_values = [ i*self.phi_spacing for i in range(self.num_phi) ]
    
        self.q_spacing = q_spacing
        self.q_min = np.min( detector.recpolar[:,0] )
        self.q_max = np.max( detector.recpolar[:,0] )
        self.q_values = np.arange(self.q_min, self.q_max, self.q_spacing)
        self.num_q = len(self.q_values)
                
        self.num_datapoints = num_phi * num_q
        
        # generate the polar grid to interpolate onto
        interpoldata = np.zeros((self.num_datapoints, 3))
        interpoldata[:,0] = np.repeat(self.q_values, self.num_phi)
        interpoldata[:,1] = np.tile(self.phi_values, self.num_q)
        
        # generate a cubic interpolation from the measured intensities
        x = detector.real[:,0]
        y = detector.real[:,1]
        interpf = interpolate.interp2d(x, y, intensities, kind='cubic')
        
        # evaluate the interpolation on our polar grid
        polar_x = interpoldata[:,0] * np.cos(interpoldata[:,1])
        polar_y = interpoldata[:,0] * np.sin(interpoldata[:,1])
        interpoldata[:,2] = ( interpf(polar_x, polar_y) ).flatten()
        
        return interpoldata
        
        
    def _mask_missing(self):
        """
        somehow, we need to process the data to either skip regions of 
        poor/absent signal, or interpolate the data inside them.
        """
        
        #self._masked_coords = []
        #self.intensities = np.ma(self.intensities, mask=mask)
        
        pass # for now...
       
        
    def _unmask_missing(self):
        raise NotImplementedError()
        
        
    def _nearest_q(self, q):
        """
        Get the value of q nearest to the argument that is on our computed grid.
        """
        if (q % self.q_spacing == 0.0) and (q > self.q_min) and (q < self.q_max):
            pass
        else:
            q = self.q_values[ bisect_left(self.q_values, q) ]
            logger.warning('Passed value `q` not on grid -- using closest '
                           'possible value')
        return q
        
        
    def _nearest_phi(self, phi):
        """
        Get value of phi nearest to the argument that is on our computed grid.
        """
        if (phi % self.phi_spacing == 0.0):
            pass
        else:
            phi = self.phi_values[ bisect_left(self.phi_values, phi) ]
            logger.warning('Passed value `phi` not on grid -- using closest '
                           'possible value')
        return phi
        
        
    def _nearest_delta(self, delta):
        """
        Get the value of delta nearest to a multiple of phi_spacing.
        
        This is really just the same as _nearest_phi() right now (BAD)
        """
        if (delta % self.phi_spacing == 0.0):
            pass
        else:
            pdeltahi = self.phi_values[ bisect_left(self.phi_values, delta) ]
            logger.warning('Passed value `delta` not on grid -- using closest '
                           'possible value')
        return delta
        
        
    def _intensity_index(self, q, phi):
        """
        Returns the index of self.intensities that matches the passed values
        of `q` and `phi`.
        
        Parameters
        ----------
        q : float
            The magnitude of the scattering vector.
        phi : float
            The azimuthal angle.
        
        Returns
        -------
        index : int
            The index of self.intensities that is closest to the passed values,
            such that self.intensities[index,2] -> I(q,phi)
        """
        
        q = self._nearest_q(q)
        phi = self._nearest_phi(phi)
        
        index = ( int(q/self.q_spacing) - int(self.q_min/self.q_spacing) ) + \
                    int(phi/self.phi_spacing)
        
        return index
        
    
    def I(self, q, phi):
        """
        Return the intensity a (q,phi).
        """
        return self.intensities[self._intensity_index(q,phi),2]
        
        
    def qintensity(self, q):
        """
        Averages over the azimuth phi to obtain an intensity for magnitude |q|.
        
        Parameters
        ----------
        q : float
            The scattering vector magnitude to calculate the intensity at.
        
        Returns
        -------
        intensity : float
            The intensity at |q| averaged over the azimuth : < I(|q|) >_phi.
        """
        
        q = self._nearest_q(q)
        
        ind = np.where(self.intensities[:,0] == q)
        intensity = np.mean( self.intensities[ind,2] )
        
        return intensity
        
        
    def intensity_profile(self):
        """
        Averages over the azimuth phi to obtain an intensity profile.
        
        Returns
        -------
        intensity_profile : ndarray, float
            An n x 2 array, where the first dimension is the magnitude |q| and
            the second is the average intensity at that point < I(|q|) >_phi.
        """
                
        intensity_profile = np.zeros((self.num_q, 2))
        
        for i,q in enumerate(self.q_values):
            intensity_profile[i,0] = q
            intensity_profile[i,1] = self.qintensity(q)

        return intensity_profile
        
        
    def intensity_maxima(self, smooth_strength=30.0):
        """
        Find the positions where the intensity profile is maximized.
        
        Parameters
        ----------
        smooth_strength : float
            Controls the strength of the smoothing function used to make sure
            that noise is not picked up as a local maxima. Increase this value
            if you're picking up noise as maxima, decrease it if you're missing
            maxima.
        
        Returns
        -------
        maxima : list of floats
            A list of the |q| positions where the intensity is maximized.
        """
        
        # first, smooth; then, find local maxima based on neighbors
        intensity = self.intensity_profile()
        a = utils.smooth(intensity, beta=smooth_strength)
        maxima = np.where(np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True] == True)[0]
        
        return maxima
        
        
    def correlate(self, q1, q2, delta):
        """
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi.
        
        Parameters
        ----------
        q1 : float
            The magnitude of the first position to correlate.
        
        q2 : float
            The magnitude of the second position to correlate.
            
        delta : float
            The angle between the first and second q-vectors to calculate the
            correlation for.
            
        Returns
        -------
        correlation : float
            The correlation between q1/q2 at angle delta. Specifically, this
            is the correlation function  with the mean subtracted <x*y> - <x><y>
            
        See Also
        --------
        odin.xray.Shot.correlate_ring
            Correlate for many values of delta
        """
        
        # we need to assume that previously the interpolation function gave
        # us a regular spaced azimuthal angular coordinate
        delta = self._closest_delta(delta)
        
        correlation = 0.0
        mean1 = 0.0
        mean2 = 0.0
        
        for phi in self.phi_values:
            
            if (  (q1,phi) not in self._masked_coords ) and (  (q2,phi+delta) not in self._masked_coords ):            
                x = self.I(q1, phi)
                y = self.I(q2, phi+delta)
                mean1 += x
                mean2 += y
                correlation += x*y
        
        correlation /= float(self.num_phi)
        mean1 /= float(self.num_phi)
        mean2 /= float(self.num_phi)
        
        return correlation - (mean1*mean2)
        
        
    def correlate_ring(self, q1, q2):
        """
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi, for many values
        of delta.
        
        Parameters
        ----------
        q1 : float
            The magnitude of the first position to correlate.
        
        q2 : float
            The magnitude of the second position to correlate.
            
        Returns
        -------
        correlation_ring : ndarray, float
            A 2d array, where the first dimension is the value of the angle
            delta employed, and the second is the correlation at that point.
            
        See Also
        --------
        odin.xray.Shot.correlate
            Correlate for one value of delta
        """
        
        # recall the possible deltas are really the possible values of phi
        correlation_ring = np.zeros(( self.num_phi, 2 ))
        correlation_ring[:,0] = np.array(self.phi_values)
        
        # now just correlate for each value of delta
        for i in range(self.num_phi):
            correlation_ring[i,1] = self.correlate(q1, q2, correlation_ring[i,0])
        
        return correlation_ring
        
    
class Shotset(Shot):
    """
    A collection of xray 'shots', and methods for anaylzing statistical
    properties across shots.
    
    See Also
    --------
    odin.xray.Shot
        A single shot dataset.
    """
        
    def __init__(self, list_of_shots):
        """
        Generate a Shotset instance, representing a collection of Shot objects.
        
        Parameters
        ----------
        list_of_shots : list of odin.xray.Shot objects
            The shots to include in a set. Should be related, i.e. all the same
            experimental system. All of the shots should be projected onto
            the same (q,phi) grid (using odin.xray.Shot._interpolate_to_polar),
            which should be done by default. If not, the class will try to fix
            things and whine if unsuccessful.
            
        See Also
        --------
        odin.xray.Shot
            A single shot dataset.
        """
        
        self.shots = list_of_shots
        self.num_shots = len(list_of_shots)
    
        self._check_qvectors_same()
        
        
    def _check_qvectors_same(self):
        """
        For a lot of the below to work, we need to make sure that all of the 
        q-phi grids in the shots are the same. If some are different, here
        we recalculate them.
        """
        raise NotImplementedError()
    
    
    def I(self, q, phi):
        """
        Return the intensity at (q,phi).
        """
        
        intensity = 0.0
    
        for shot in self.shot_list:
            intensity += shot.I(q,phi)
            
        intensity /= float(self.num_shots)
        
        return intensity 


    def qintensity(self, q):
        """
        Averages over the azimuth phi to obtain an intensity for magnitude |q|.

        Parameters
        ----------
        q : float
            The scattering vector magnitude to calculate the intensity at.

        Returns
        -------
        intensity : float
            The intensity at |q| averaged over the azimuth : < I(|q|) >_phi.
        """
        
        qintensity = 0.0
    
        for shot in self.shot_list:
            qintensity += shot.qintensity(q)
            
        qintensity /= float(self.num_shots)
        
        return qintensity
        
    
    def intensity_profile(self):
        """
        Averages over the azimuth phi to obtain an intensity profile.

        Returns
        -------
        intensity_profile : ndarray, float
            An n x 2 array, where the first dimension is the magnitude |q| and
            the second is the average intensity at that point < I(|q|) >_phi.
        """
        
        intensity_profile = shot_list[0].intensity_profile()
    
        for shot in self.shot_list[1:]:
            intensity_profile[:,1] += shot.intensity_profile()
            
        intensity_profile[:,1] /= float(self.num_shots)
        
        return intensity_profile
        
        
    def correlate(self, q1, q2, delta):
        """
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi.

        Parameters
        ----------
        q1 : float
            The magnitude of the first position to correlate.

        q2 : float
            The magnitude of the second position to correlate.

        delta : float
            The angle between the first and second q-vectors to calculate the
            correlation for.

        Returns
        -------
        correlation : float
            The correlation between q1/q2 at angle delta. Specifically, this
            is the correlation function  with the mean subtracted <x*y> - <x><y>

        See Also
        --------
        odin.xray.Shot.correlate_ring
            Correlate for many values of delta
        """
        
        correlation = 0.0
    
        for shot in self.shot_list:
            correlation += shot.correlate(q1, q2, delta)
            
        correlation /= float(self.num_shots)
        
        return correlation
        
        
    def correlate_ring(self, q1, q2):
        """
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi, for many values
        of delta.

        Parameters
        ----------
        q1 : float
            The magnitude of the first position to correlate.

        q2 : float
            The magnitude of the second position to correlate.

        Returns
        -------
        correlation_ring : ndarray, float
            A 2d array, where the first dimension is the value of the angle
            delta employed, and the second is the correlation at that point.

        See Also
        --------
        odin.xray.Shot.correlate
            Correlate for one value of delta
        """   
    
        # just average all the correlation_rings from each shot
        correlation_ring = shot_list[0].correlate_ring(q1, q2)
    
        for shot in self.shot_list[1:]:
            correlation_ring[:,1] += shot.correlate_ring(q1, q2)
            
        correlation_ring[:,1] /= float(self.num_shots)
        
        return correlation_ring

    
def simulate_shot(traj, num_molecules, beam, detector,
                  traj_weights=None, force_no_gpu=False):
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
    traj : odin.mdtraj
        A trajectory object that contains a set of structures, representing
        the Boltzmann ensemble of the sample. If len(traj) == 1, then we assume
        the sample consists of a single homogenous structure, replecated 
        `num_molecules` times.
        
    detector : odin.xray.Detector
        A detector object the shot will be projected onto.
        
    beam : odin.xray.beam
        A descriptor of the beam used in the 'experiment'.
        
    num_molecules : int
        The number of molecules estimated to be in the `beam`'s focus.
        
    traj_weights : ndarray, float
        If `traj` contains many structures, an array that provides the Boltzmann
        weight of each structure. Default: if traj_weights == None, weights
        each structure equally.
        
    force_no_gpu : bool
        Run the (slow) CPU version of this function.
        
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
    

    
    pass
    
    
    
    
