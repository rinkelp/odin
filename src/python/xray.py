
"""
Classes, methods, functions for use with xray scattering experiments.
"""

import numpy as np

import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# FUNDAMENTAL CONSTANTS

h = 4.135677516e-15   # Planks constant | eV s
c = 299792458     # speed of light  | m / s

# ------------------------------------------------------------------------------


class Beam(object):
    """
    Class that converts energies, wavelengths, frequencies, and wavenumbers.
    """
    
    def __init__(self, **kwargs):
        
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
        
        
def generate_test_detector(spacing=0.02, lim=1.0, k=4.33, l=1.0):
    """
    Generates a simple grid detector that can be used for testing.
    
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
    
    beam = Beam(wavenumber=k)
    
    x = np.arange(-lim, lim, spacing)
    xx, yy = np.meshgrid(x, x)
    
    xyz = np.zeros((len(x)**2, 3))
    xyz[:,0] = xx.flatten()
    xyz[:,1] = yy.flatten()
    
    np.savetxt('test_detector.dat', xyz)
    
    detector = Detector('test_detector.dat', l, beam)
        
    return detector
    