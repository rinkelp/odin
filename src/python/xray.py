# THIS FILE IS PART OF ODIN


"""
Classes, methods, functions for use with xray scattering experiments.
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

import cPickle
from bisect import bisect_left
import multiprocessing as mp
from threading import Thread

import numpy as np
from scipy import interpolate, fftpack, weave, special
from scipy.ndimage import filters
from scipy.misc import factorial

from odin.math import arctan3, rand_pairs, smooth
from odin import cpuscatter
from odin.bcinterp import Bcinterp
from odin.refdata import cromer_mann_params

from mdtraj import trajectory, io

# try to import the gpuscatter module
GPU = True
try:
    from odin import gpuscatter
except ImportError as e:
    logger.debug('Could not find `gpuscatter` module, proceeding without it.'
                 ' Note that this may break some functionality!')
    GPU = False


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
                self.energy = self.wavenumber * h * c * 10.**7. / (2.0 * np.pi)
                
            elif key == 'wavelength':
                self.wavelength = float(kwargs[key])
                self.energy = h * c * 10.**7. / self.wavelength
                
            elif key == 'frequency':
                self.frequency = float(kwargs[key])
                self.energy = self.frequency * h
                
            else:
                raise ValueError('%s not a recognized kwarg' % key)
        
        # perform the rest of the conversions
        self.wavelength = h * c * 10.**7. / self.energy
        self.wavenumber = 2.0 * np.pi / self.wavelength
        self.frequency = self.energy * (1000. / h)
        
        # some aliases
        self.k = self.wavenumber
    
    
class Detector(Beam):
    """
    Class that provides a plethora of geometric specifications for a detector
    setup. Also provides loading and saving of detector geometries.
    """
    
    def __init__(self, xyz, path_length, k, xyz_type='explicit', 
                 coord_type='cartesian'):
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
        xyz : ndarray (OR sometimes list, see xyz_type kwarg)
            An n x 3 array specifying the (x,y,z) positions of each pixel.
            
        path_length : float
            The distance between the sample and the detector, in the same units
            as the pixel dimensions in the input file.
            
        k : float or odin.xray.Beam
            The wavenumber of the incident beam to use. Optionally A Beam 
            object, defining the beam energy.
            
        Optional Parameters
        -------------------            
        xyz_type : str, {'explicit', 'implicit'}
            Choose between an explicit and implicit detector type. If explicit,
            then `xyz` is an array explicitly definining the xyz coordinates of
            each pixel. If implicit, then `xyz` is a `grid_list` that 
            implicitly specifies the detector pixels. See the factor function 
            Detector.from_basis() for extensive documentation of this feature,
            and a handle for constructing detectors of this type.
            
        coord_type : str {'cartesian', 'polar'}
            Indicate the most natural representation of your detector. This
            only affects downstream performance, not behavior. If you don't know,
            choose 'cartesian'.
        """
        
        if xyz_type == 'explicit':
            
            if type(xyz) != np.ndarray:
                raise ValueError('Explicit xyz pixels required, `xyz` must be np.ndarray')
            
            self._pixels = xyz
            self._grid_list = None
            
            self.num_pixels = xyz.shape[0]
            self._xyz_type = 'explicit'
                
        elif xyz_type == 'implicit':
            
            # I am being a little tricky here -- by passing the grid_list into
            # the xyz argument. The safety of this move is predicated on
            # the user employing the xyz_type kwarg appropriately. Hopefully 
            # this doesn't cause too much trouble later... -- TJL
            
            if type(xyz) != list:
                raise ValueError('Implicit xyz pixels required, `xyz` must be list')
            self._xyz_type = 'implicit'
            
            self._grid_list = xyz
            for g in self._grid_list:
                if (len(g) != 3) or (type(g) != tuple):
                    logger.critical("self._grid_list: %s" % str(self._grid_list))
                    raise ValueError('grid list not 3-tuple')
            
            self.num_pixels = int(np.sum([ b[1][0]*b[1][1]*b[1][2] for b in self._grid_list ]))
            self._pixels    = None
            
        else:
            raise ValueError("`xyz_type` must be one of {'explicit', 'implicit'}")
        
        
        # parse wavenumber
        if isinstance(k, Beam):
            self.k = k.wavenumber
        elif type(k) in [float, np.float64, np.float32]:
            self.k = k
        else:
            raise ValueError('`k` must be a float or odin.xray.Beam')
        
        if coord_type in ['cartesian', 'polar']:
            self.coord_type = coord_type
        else:
            raise ValueError("`coord_type` must be one of {'cartesian', 'polar'}")
        
        self.path_length = path_length
        
        
    @classmethod
    def from_basis(cls, grid_list, path_length, k):
        """
        Factory function - generate a detector object from an implicit "basis"
        scheme, where the detector is a set of rectangular grids specified by
        an x/y/z-spacing and size, rather than explicit x/y/z coords.
        
        This method provides optimal memory performance. It is slightly more
        abstract to instantiate, but if your data conforms to this kind of
        representation, we recommend using it (especially for large datasets).
        
        Parameters
        ----------
        grid_list : list
            A list of detector arrays, specificed "implicitly" by the following
            scheme: 
            
                -- A corner position (top left corner)
                -- x-basis, y-basis, z-basis vectors, aka the pixel spacing
                -- array shape (detector dimensions)
             
            This method then generates a detector object by building a grid 
            specifed by these parameters - e.g. if the x/y-basis vectors are
            both unit vectors, and shape is (10, 10), you'd get an xyz array
            with pixels of the form (0,0), (0,1), (0,2), ... (10,10).
            
            The `grid_list` argument then should be a list-of-tuples of the form
            
                grid_list = [ ( basis, shape, corner ) ]
            
            where
            
                basis   : tuple of floats (x_basis, y_basis, z_basis)
                shape   : tuple of floats (x_size, y_size, z_size)
                corner  : tuple of floats (x_corner, y_corner, z_corner)
            
            Each item in the list gets instatiated as a different detector
            "array", but all are included into the same detector object.
            
        path_length : float
            The distance between the sample and the detector, in the same units
            as the pixel dimensions in the input file.

        k : float or odin.xray.Beam
            The wavenumber of the incident beam to use. Optionally A Beam 
            object, defining the beam energy.
        """
        if type(grid_list) == tuple: # be generous...
            grid_list = [grid_list]
        elif type(grid_list) == list:
            if (not type(grid_list[0] == tuple)) and (len(grid_list[0]) == 3):
                raise ValueError('grid_list must be a list of 3-tuples')
        else:
            raise ValueError('grid_list must be a list of 3-tuples')
        d = Detector(grid_list, path_length, k, xyz_type='implicit')
        return d
      
        
    def implicit_to_explicit(self):
        """
        Convert an implicit detector to an explicit one (where the xyz pixels
        are stored in memory).
        """
        if not self.xyz_type == 'implicit':
            raise Exception('Detector must have xyz_type implicit for conversion.')
        self._pixels = self.xyz
        self._xyz_type = 'explicit'
        return
        
             
    @property
    def xyz_type(self):
        return self._xyz_type
        
        
    @property
    def xyz(self):
        if self.xyz_type == 'explicit':
            return self._pixels
        elif self.xyz_type == 'implicit':
            return np.concatenate( [self._grid_from_implicit(*t) for t in self._grid_list] )
        
        
    def _grid_from_implicit(self, basis, size, corner):
        """
        From an x, y, z basis vector set, and the dimensions of the grid,
        generates an explicit xyz grid.
        """
        
        assert self.xyz_type == 'implicit'
        
        if basis[0] != 0.0:
            x_pixels = np.arange(0, basis[0]*size[0], basis[0])[:size[0]]
        else:
            x_pixels = np.zeros( size[0] )
        assert len(x_pixels) == size[0]
        
        if basis[1] != 0.0:
            y_pixels = np.arange(0, basis[1]*size[1], basis[1])[:size[1]]
        else:
            y_pixels = np.zeros( size[1] )
        assert len(y_pixels) == size[1]
        
        if basis[2] != 0.0:
            z_pixels = np.arange(0, basis[2]*size[2], basis[2])[:size[2]]
        else:
            z_pixels = np.zeros( size[2] )
        assert len(z_pixels) == size[2]
        
        x = np.repeat(x_pixels, size[1]*size[2])
        y = np.tile( np.repeat(y_pixels, size[2]), size[0] )
        z = np.tile(z_pixels, size[0]*size[1])

        xyz = np.vstack((x, y, z)).transpose()
        xyz += np.array(corner)
        
        assert xyz.shape[0] == size[0] * size[1] * size[2]
        assert xyz.shape[1] == 3
        
        return xyz
        
        
    def _grid_from_gridlist(grid_list):
        """
        Transforms a gridlist (list of tuples describing grided detectors) into
        an xyz representation of the detctor, aka the explicit coordiantes of each 
        pixel.

        Parameters
        ----------
        grid_list: list of tuples
            A basis vector representation of the detector pixels
                grid_list = [ ( basis, shape, corner ) ]

        Returns
        -------
        xyz : ndarray, float, 3D
            An n x 3 array of the coordinates of each pixel.
        """
        xyz_list = []
        for g in grid_list:
            xyz_list.append( grid_from_implicit(*g) )
        return np.vstack(xyz_list)
        
        
    @property
    def real(self):
        real = self.xyz.copy()
        real[:,2] += self.path_length
        return real
        
        
    @property
    def polar(self):
        return self._real_to_polar(self.real)
        
        
    @property
    def reciprocal(self):
        return self._real_to_reciprocal(self.real)
        
        
    @property
    def recpolar(self):
        a = self._real_to_recpolar(self.real)
        a[:,1] = 0.0 # convention, re: Dermen, TJL todo, don't think this is right... (flat ewald sphere)
        return a
        
        
    def _real_to_polar(self, xyz):
        """
        Convert the real-space representation to polar coordinates.
        """
        polar = self._to_polar(xyz)
        return polar
        
        
    def _real_to_reciprocal(self, xyz):
        """
        Convert the real-space to reciprocal-space in cartesian form.
        """
        
        assert len(xyz.shape) == 2
        assert xyz.shape[1] == 3
        
        # generate unit vectors in the pixel direction, origin at sample
        S = self.real.copy()
        S = self._unit_vector(S)
        
        # generate unit vectors in the z-direction
        S0 = np.zeros(xyz.shape)
        S0[:,2] = np.ones(xyz.shape[0])
        
        assert S.shape == S0.shape
        
        q = self.k * (S - S0)
        
        return q
        
        
    def _real_to_recpolar(self, xyz):
        """
        Convert the real-space to reciprocal-space in polar form, that is
        (|q|, theta , phi).
        """
        reciprocal_polar = self._to_polar( self._real_to_reciprocal(xyz) )
        return reciprocal_polar
        
        
    def _reciprocal_to_real(self, qxyz):
        """
        Converts a q-space cartesian representation (`qxyz`) into a real-space
        cartesian representation (`xyz`)
        """
        
        assert len(qxyz.shape) == 2
        xyz = np.zeros_like(qxyz)
        
        q_mag = self._norm(qxyz)
        q_mag[q_mag == 0.0] = 1.0e-300
        
        h = self.path_length * np.tan( 2.0 * np.arcsin( qxyz[:,2] / q_mag ) )
        q_xy = np.sqrt( np.sum( np.power(qxyz[:,:2], 2), axis=1 ) )
        q_xy[q_xy == 0.0] = 1.0e-300
        hn = ( h / q_xy )
        
        for i in range(2):
            xyz[:,i] = qxyz[:,i] * hn

        xyz[:,2] = self.path_length
        xyz = xyz[::-1,:]  # for consistency with other methods
        
        return xyz
        
        
    def _norm(self, vector):
        """
        Compute the norm of an n x m array of vectors, where m is the dimension.
        """
        if len(vector.shape) == 2:
            assert vector.shape[1] == 3
            norm = np.sqrt( np.sum( np.power(vector, 2), axis=1 ) )
        elif len(vector.shape) == 1:
            assert vector.shape[0] == 3
            norm = np.sqrt( np.sum( np.power(vector, 2) ) )
        else:
            raise ValueError('Shape of vector wrong')
        return norm
        
        
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
        for i in range(vector.shape[0]):
            unit_vectors[i,:] = vector[i,:] / norm[i]
        
        return unit_vectors
        
        
    def _to_polar(self, vector):
        """
        Converts n m-dimensional `vector`s to polar coordinates. By polar
        coordinates, I mean the cannonical physicist's (r, theta, phi), no
        2-theta business.
        """
        
        polar = np.zeros( vector.shape )
        
        polar[:,0] = self._norm(vector)
        polar[:,1] = np.arccos(vector[:,2] / (polar[:,0]+1e-16)) # cos^{-1}(z/r)
        polar[:,2] = arctan3(vector[:,1], vector[:,0])     # y first!
        
        return polar
        
        
    @classmethod
    def generic(cls, spacing=1.00, lim=100.0, energy=10.0, flux=100.0, l=50.0, 
                force_explicit=False):
        """
        Generates a simple grid detector that can be used for testing
        (factory function). 

        Optional Parameters
        -------------------
        spacing : float
            The real-space grid spacing
        lim : float
            The upper and lower limits of the grid
        k : float
            Wavenumber of the beam
        l : float
            The path length from the sample to the detector, in the same units
            as the detector dimensions.

        Returns
        -------
        detector : odin.xray.Detector
            An instance of the detector that meets the specifications of the 
            parameters
            
            
        Optional Parameters
        -------------------
        force_explicit : bool
            Forces the detector to be xyz_type explicit. Mostly for debugging.
            Recommend keeping `False`.
        """
        
        beam = Beam(flux, energy=energy)

        if not force_explicit:
            
            basis = (spacing, spacing, 0.0)
            dim = int(2*(lim / spacing) + 1)
            shape = (dim, dim, 1)
            corner = (-lim, -lim, 0.0)
            basis_list = [(basis, shape, corner)]
            
            detector = Detector.from_basis(basis_list, l, beam)

        else:
            x = np.arange(-lim, lim+spacing, spacing)
            xx, yy = np.meshgrid(x, x)

            xyz = np.zeros((len(x)**2, 3))
            xyz[:,0] = yy.flatten()
            xyz[:,1] = xx.flatten()

            detector = Detector(xyz, l, beam)

        return detector
        
        
    @classmethod
    def generic_polar(cls, q_spacing=0.02, q_lim=5.0, q_values=None,
                       angle_spacing=1.0, energy=10.0, flux=100.0, l=50.0):
        """
        Generates a simple grid detector that can be used for testing
        (factory function). 

        Optional Parameters
        -------------------
        q_spacing : float
            The |q| grid spacing
                        
        q_lim : float
            The upper limit of the q-vector
            
        q_values : ndarray, float
            Instead of the spacing, you can include a set of specific |q| values
            to produce specific rings. If not `None`, this list over-rides
            the `q_spacing` and `q_lim` arguments.
            
        angle_spacing : float
            The angular grid spacing, *in degrees*
            
        k : float
            Wavenumber of the beam
            
        l : float
            The path length from the sample to the detector, in the same units
            as the detector dimensions.

        Returns
        -------
        detector : odin.xray.Detector
            An instance of the detector that meets the specifications of the 
            parameters
        """

        if q_values == None:
            q_values = np.arange(0.0, q_lim, q_spacing)
            
        phi_values = np.arange(0.0, 2.0*np.pi, angle_spacing*(2.0*np.pi/360.0) )
            
        polar = np.zeros(( len(q_values) * len(phi_values), 2 ))
        polar[:,0] = np.repeat(q_values, len(phi_values))
        polar[:,1] = np.tile(phi_values, len(q_values))

        xyz = np.zeros(( polar.shape[0], 3 ))
        xyz[:,0] = polar[:,0] * np.cos(polar[:,1])
        xyz[:,1] = polar[:,0] * np.sin(polar[:,1])
        
        beam = Beam(flux, energy=energy) 

        detector = Detector(xyz, l, beam, coord_type='polar')

        return detector
        
        
    def _to_serial(self):
        """ serialize the object to an array """
        s = np.array( cPickle.dumps(self) )
        s.shape=(1,) # a bit nasty...
        return s
        
        
    @classmethod
    def _from_serial(self, serialized):
        """ recover a Detector object from a serialized array """
        if serialized.shape == (1,):
            serialized = serialized[0]
        d = cPickle.loads( str(serialized) )
        return d
    
        
    def save(self, filename):
        """
        Writes the current Detector to disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file to save.
        """
                
        if not filename.endswith('.dtc'):
            filename += '.dtc'
        
        io.saveh(filename, detector=self._to_serial())
        logger.info('Wrote %s to disk.' % filename)

        return
        
        
    @classmethod
    def load(cls, filename):
        """
        Loads the a Detector from disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file.

        Returns
        -------
        shotset : odin.xray.Shotset
            A shotset object
        """
        
        if not filename.endswith('.dtc'):
            raise ValueError('Must load a detector file (.dtc extension)')
        
        hdf = io.loadh(filename)
        d = Detector._from_serial(hdf['detector'])        
        return d


class ImageFilter(object):
    """
    A pre-processor class that provides a set of 'filters' that generally
    improve the x-ray image data. The complete list of possibilities includes,
    along with the function name that implements that functionality in (.):
        
        -- removing `hot` outlier pixels   (hot_pixels)
        -- correcting for polarization     (polarization)
        -- remove edges around ASICs       (mask_edges)
        
    The way this class works is that you generate an ImageFilter object, 'turn
    on' the features from the list above you want, and then apply that filter
    to some raw intensity data.
    
    Example
    -------
    >>> flt = ImageFilter()          # initialize class
    >>> flt.hot_pixels(abs_std=3.0)  # mask pixels more than 3 STD from the mean
    >>> flt.polarization(0.99)       # remove polarization
    >>>
    >>> # now apply the filter to some data
    >>> new_intensities1, mask1 = flt(intensities1)
    >>> new_intensities2, mask2 = flt(intensities2)
        
    For ease, the ImageFilter initialization method also takes kwargs that
    instantiate a class with some filters already applied. Eg. the above filter
    is equivalent to
        
    >>> flt = ImageFilter(abs_std=3.0, polarization=0.99)
    """
        
    # Note to programmer: here is how this class works. Each method consists
    # of two parts, one public that "turns on" the method for the filter
    # and one private that is called _apply_<filtername> that actually does
    # the calcuation that makes the filter happen. The public method should
    # store all necessary parameters as private attributes, append its name
    # as a string to self._methods_to_apply. The private method should use
    # those parameters and do the calculation.
    #
    # Also don't forget to add your method to the self.apply() method, which
    # checks self._methods_to_apply for what to do.
        
    def __init__(self, abs_std=None, polarization=None, border_pixels=None):
        """
        Initialize an image filter. Optional kwargs initialize the filter with
        some standard filters activated. Otherwise, you have to turn each filter
        on one-by-one.
        
        Optional Parameters
        -------------------
        abs_std : float
            Filter out any pixel that is further than `abs_std` * STD from the
            mean (i.e. this an n-sigma cutoff).
        
        polarization : tuple (float, odin.xray.detector)
            A 2-tuple of the polarization factor to apply to the data, and the
            detector to employ.
        
        border_pixels : int
            Filters this number of pixels around the border of each detector
            ASIC. ASICs are detected automatically.
        """
        
        self._methods_to_apply = list()
        
        # parse the kwargs and turn them into filters
        if abs_std:
            self.hot_pixels(abs_std)
        if polarization:
            self.polarization(*polarization)
        if border_pixels:
            self.detector_mask(border_pixels)
            
        return
        
        
    def __call__(self, intensities, intensities_shape=None):
        """
        Apply the `ImageFilter` to `intensities`. An alias for self.apply().
        
        Parameters
        ----------
        intensities : ndarray, float
            The intensity data.
        
        Optional Parameters
        -------------------
        intensities_shape : two-tuple
            The shape of the intensities (a 2-d array). If passed, the
            intensities array will be re-shaped to this shape.
        """
        return self.apply(intensities, intensities_shape=intensities_shape)
        
        
    def apply(self, intensities, intensities_shape=None):
        """
        Apply the `ImageFilter` to `intensities`.
        
        Parameters
        ----------
        intensities : ndarray, float
            The intensity data.
            
        Optional Parameters
        -------------------
        intensities_shape : two-tuple
            The shape of the intensities (a 2-d array). If passed, the
            intensities array will be re-shaped to this shape.
        """
        
        # we'll guarentee flat array, but make available the shape if necessary
        if len(intensities.shape) == 1:
            if intensities_shape == None:
                raise ValueError('If `intensities` is flat, you must also '
                                 'provide the `intensities_shape` argument.')
            else:
                self._intensities_shape = intensities_shape
        else:
            self._intensities_shape = intensities.shape
            intensities = intensities.flatten()
        
        # initialize mask
        mask = np.zeros(self._intensities_shape, dtype=np.bool)
        
        # iterate through each filter method, if it's called for apply it
        if 'hot_pixels' in self._methods_to_apply:
            intensities, mask = self._apply_hot_pixels(intensities, mask)
        
        if 'polarization' in self._methods_to_apply:
            intensities = self._apply_polarization(intensities, self._detector)
        
        if 'detector_mask' in self._methods_to_apply:
            intensities, mask = self._apply_detector_mask(intensities, mask)

        return intensities, mask
        
        
    def hot_pixels(self, abs_std=3.0):
        """
        Filter out any pixel that is further than `abs_std` * STD from the
        mean (i.e. this an n-sigma cutoff).
        
        Parameters
        ----------
        abs_std : float
            The STD multiplication factor that sets the strength of the filter
        """
        self._abs_std = abs_std
        self._methods_to_apply.append('hot_pixels')
        return
        
        
    def _apply_hot_pixels(self, i, mask):
        """
        Apply the hot pixel filter to `i`
        """
        i = i.reshape(self._intensities_shape)
        cutoff = self._abs_std * i.std()
        mask[ np.abs(i - i.mean()) > cutoff ] = np.bool(True)
        return i, mask
        
        
    def polarization(self, polarization_factor, detector):
        """
        Remove the effects of beam polarization on the detected intensities.
        
        Parameters
        ----------
        polarization_factor : float
            
        """
        
        # TJL todo : is passing a detector really the best way (most user 
        # friendly) to do this?
        
        self._thetas = detector.polar[:,1].copy() / 2.0 # this is the crystallographic theta
        self._phis   = detector.polar[:,2].copy()
        self._polarization_factor = polarization_factor
        self._methods_to_apply.append('polarization')
        self._detector = detector
        
        return
        
        
    def _apply_polarization(self, i, mask):
        """
        Apply the polarization filter to `i`
        """
        cf  = ( self._polarization_factor ) * ( 1.0 - np.power( np.sin(self._thetas) * np.cos(self._phis), 2 ))
        cf += ( 1.0 - self._polarization_factor ) * ( 1.0 - np.power( np.sin(self._thetas) * np.sin(self._phis), 2 ))
        i = i.flatten()
        i /= cf
        return i

        
    def detector_mask(self, border_pixels=1, num_bins=1e5, beta=1.0):
        """
        Mask the detector mask and edges of each ASIC, to a width of 
        `border_pixels`. These border pixels are sometimes likely to spike.
        
        Parameters
        ----------
        border_pixels : int
            The width of the border around each ASIC to mask, in pixel units.
            
        Additional Parameters
        ---------------------
        num_bins : int
            The number of bins to use in the histogram-based image segmentation 
            used to find the mask regions. In tests, 10k worked well. If the
            detector mask is not getting separated from the measured pixels,
            increase this number.
            
        beta : float
            The smoothing parameter used to smooth the histogram-based image 
            segmentation. If the detector mask is not getting separated from the
            measured pixels, lower this number. beta=10.0 worked well in tests.
        """
        self._border_pixels = border_pixels
        self._num_bins = num_bins
        self._beta = beta
        self._methods_to_apply.append('detector_mask')
        return
        
        
    def _apply_detector_mask(self, i, mask):
        """
        Apply the edge mask filter to `i`, the intensities.
        """
        dmask = self._find_detector_mask(i, num_bins=self._num_bins, 
                                         beta=self._beta)
        mask += self._mask_border(dmask, self._border_pixels)
        return i, mask
        
        
    def _find_detector_mask(self, intensities, num_bins=1e5, beta=1.0):
        """
        Find the detector regions by histogramming. The histogram of intensities 
        should be bi-modal, so we need to split it into two groups. To do this,
        smooth and choose the first local minimum.
        """
        
        # reshape intensties for convienence
        intensities = intensities.reshape(self._intensities_shape)
        
        mask = np.zeros(intensities.shape, dtype=np.bool)
        hist, bin_edges = np.histogram( intensities, bins=num_bins )
        
        a = smooth(hist, beta=beta)
        minima = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True] == True)[0]
        cutoff = a[minima[0]]
        
        logger.debug("Mask cutoff: %f" % cutoff)
        mask[ intensities < cutoff ]  = np.bool(True)
        mask[ intensities >= cutoff ] = np.bool(False)
        
        mask = mask.reshape(self._intensities_shape) # ensure mask is square
        
        return mask
        
        
    def _mask_border(self, mask, border_pixels):
        """
        Returns a version of `mask` with an additional `border_pixels` masked
        around the masked region.
        """
        
        if border_pixels < 1:
            logger.warning('_mask_border() called with `border_pixels`=%d' % border_pixels)
            return mask
        
        if len(mask.shape) == 1:
            logger.critical("Mask shape: %s" % str(mask.shape))
            raise ValueError('mask must be two-dimensional')
        
        # recall : in an array np.bool(True) + np.bool(True) = np.bool(True)
        for i in range(1,border_pixels+1):
            mask += np.roll(mask, i, axis=0)
            mask += np.roll(mask, -i, axis=0)
        
            mask += np.roll(mask, i, axis=1)
            mask += np.roll(mask, -i, axis=1)
        
        return mask
        
        
    def _find_detector_mask_by_region(self, intensities):
        """
        Return a mask representing the detector. This works by segmenting the
        image along sharp edges, finding the means of each segment, and 
        splitting the segments into two clusters (measured/mask) using 
        a histogramming method.
        """
        
        logger.debug("Locating detector mask regions...")
        logger.warning("Finding the mask by region is (1) slow and (2) untested"
                       ". Recommend using _find_detector_mask().")
        
        # shape-check intensities
        assert len(intensities.shape) == 2
        
        # find the detector edges, they will be 0's -- unclassified are -1's
        classified = self._find_edges(intensities) - 1.0
        
        # loop over all non-edge pixels, and classify regions by lumping
        # adjacent pixels into groups -- results in `classified`        
        def adjacent(i,j):
            """
            Return a list of the indices of the unclassified pixels adjacent 
            to (i,j).
            """
            logger.debug("Finding pixels adjacent to: %s" % str((i,j)))
            l_adj = list()
            for pixel in [ (i-1,j), (i+1,j), (i,j-1), (i,j+1) ]:
                if (pixel[0] >= 0) and (pixel[0] < classified.shape[0]):
                    if (pixel[1] >= 0) and (pixel[1] < classified.shape[1]):
                        if (classified[pixel] == -1):
                            l_adj.append( pixel )
            return l_adj
        
        # we want to classify pixels by recursively growing adjacent regions
        adjacents = list()
        while (classified == -1).sum() > 0:
            logger.debug("Unclassified remaining: %d" % (classified == -1).sum())
            
            # choose a new pixel to grow a region out of
            for i in range(intensities.shape[0]):
                for j in range(intensities.shape[1]):
                    if classified[(i,j)] == -1:
                        seed_pixel = (i,j)
                        break
                    break
                    
            logger.debug("Seed pixel: %s" % str(seed_pixel))
            
            # create a new region
            classified[seed_pixel] = classified.max() + 1
            adjacents.extend( adjacent(*seed_pixel) )
            
            # grow that region
            while len(adjacents) > 0:
                p = adjacents.pop()
                classified[p] = classified.max()
                adjacents.extend( adjacent(*p) )
                
        # next, compute the means of each region and histogram
        num_regions = classified.max() # recall zero are the edges
        means = np.zeros(num_regions)
        for i in range(1,num_regions):
            means[i] = np.mean( intensities[ classified == i ] )
            
        # histogram the means
        num_bins = num_regions/10
        if num_bins < 10: num_bins = 10
        hist, bin_edges = np.histogram(means, bins=num_bins)
        
        # the result should be bi-modal, so we need to split it into two groups
        # to do this, smooth and choose the central local minimum
        
        # todo dbl chk below
        a = smooth(hist, beta=3.0) # fiddle
        minima = np.where(np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True] == True)[0]
        cutoff = minima[1]
        
        # now generate a mask, and use the above cutoff to classify regions
        # as detector mask or not. Do this in-place on `classified`. Put
        # `1` in measured places, `0` in edges/mask locations (recall 0 was
        # previously for edges)
        for i in range(1,num_regions):
            if means[i] >= cutoff:
                classified = 1
            else:
                classified = 0
        
        # we want the mask to be true at masked positions
        classified = 1 - classified
        
        return classified
    

    def _find_edges(self, image, threshold=0.1):
        """
        Method to find the detector boundaries.
        """

        image = np.abs(filters.sobel(image, 0)) + np.abs(filters.sobel(image, 1))
        image -= image.min()

        assert image.min() == 0
        assert image.max() > 0

        logger.debug('threshold value: %d' % (image.max() * threshold))
        image = (image > (image.max() * threshold)).astype(np.int8)

        return image
        
        
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
        
    def __init__(self, intensities, detector, mask=None, interpolated_values=None):
        """
        Instantiate a Shot class.
        
        Parameters
        ----------
        intensities : ndarray, float
            A one-dimensional array of the measured intensities at each pixel
            of the detector.
        
        detector : odin.xray.Detector
            A detector object, containing the pixel positions in space.
            
        Optional Parameters
        -------------------
        interpolated_values : tuple
            If the shot has previously been interpolated onto a polar grid,
            you can instantiate it by passing

            interpolated_values = (polar_intensities, polar_mask, interp_params)

            This is mostly a function for loading saved data. The user is probably
            better off just letting the class interpolate automatically.
        """
                
        self.intensities = intensities.flatten()
        self.detector = detector
        
        if mask != None:
            if mask.shape != intensities.shape:
                raise ValueError('Mask must be same shape as intensities array!')
            self.real_mask = np.array(mask.flatten())
        else:
            self.real_mask = np.zeros(intensities.shape, dtype=np.bool)
        
        if interpolated_values == None:
            self.interpolate_to_polar() # also sets self.polar_mask
        else:
            self.polar_intensities = interpolated_values[0]
            self.polar_mask        = interpolated_values[1]
            self._unpack_interp_params(interpolated_values[2])
                    
        return
        
    @property
    def phi_values(self):
        if hasattr(self, 'phi_spacing'):
            pv = np.arange(0.0, 2.0*np.pi, self.phi_spacing)
        else:
            pv = None
        return pv
        
    @property
    def q_values(self):
        if hasattr(self, 'q_min') and hasattr(self, 'q_max') and hasattr(self, 'q_spacing'):
            qv = np.arange(self.q_min, self.q_max+self.q_spacing, self.q_spacing)
        else:
            qv = None
        return qv
        
    @property
    def num_phi(self):
        if hasattr(self, 'phi_spacing'):
            nph = len(self.phi_values)
        else:
            nph = None
        return nph
        
    @property
    def num_q(self):
        if hasattr(self, 'q_min') and hasattr(self, 'q_max') and hasattr(self, 'q_spacing'): 
            nq = len(self.q_values)
        else:
            nq = None
        return nq
        
    @property
    def num_datapoints(self):
        return self.num_phi * self.num_q
        

    def interpolate_to_polar(self, q_spacing=0.02, phi_spacing=1.0):
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
        
        Injects
        -------
        self.polar_intensities : ndarray, float
            An array of the intensities I(|q|, phi)
        """
        
        self.phi_spacing = phi_spacing * (2.0 * np.pi / 360.) # conv. to radians
        self.q_spacing = q_spacing

        mq = self.detector.recpolar[:,0]  # |q|
        self.q_min = np.min( mq )
        self.q_max = np.max( mq )

        self.polar_intensities = np.zeros(self.num_datapoints)
        self.polar_mask = np.zeros(self.num_datapoints, dtype=np.bool)
        
        # check to see what method we want to use to interpolate. Here,
        # `unstructured` is more general, but slower; implicit/structured assume
        # the detectors are grids, and are therefore faster but specific
        
        if self.detector.xyz_type == 'explicit':
            self._unstructured_interpolation()
        elif self.detector.xyz_type == 'implicit':
            self._implicit_interpolation()
        else:
            raise RuntimeError('Invalid detector passed to Shot()')
        
        self._apply_mask() # force masking of requested bad pixels
        
        
    @property
    def polar_grid(self):
        """
        Return the pixels that comprise the polar grid in (q, phi) space.
        """
        
        polar_grid = np.zeros((self.num_datapoints, 2))
        polar_grid[:,0] = np.tile(self.q_values, self.num_phi)
        polar_grid[:,1] = np.repeat(self.phi_values, self.num_q)
        
        return polar_grid
        
        
    @property
    def polar_grid_as_cart(self):
        """
        Returns the pixels that comprise the polar grid in q-cartesian space,
        (q_x, q_y)
        """
        
        pg = self.polar_grid
        pg_real = np.zeros( pg.shape )
        pg_real[:,0] = pg[:,0] * np.cos( pg[:,1] )
        pg_real[:,1] = pg[:,0] * np.sin( pg[:,1] )
        
        return pg_real
        
        
    @property
    def polar_grid_as_real_cart(self):
        """
        Returns the pixels that comprise the polar grid in real cartesian space,
        (x, y)
        """
        
        pg  = self.polar_grid
        pgr = np.zeros_like(pg)
        k = self.detector.k
        l = self.detector.path_length
        
        pg[ pg[:,0] == 0.0 ,0] = 1.0e-300
        h = l * np.tan( 2.0 * np.arcsin( pg[:,0] / (2.0*k) ) )

        pgr[:,0] = h * np.cos( pg[:,1] )
        pgr[:,1] = h * np.sin( pg[:,1] )
                
        return pgr
        
        
    def _overlap(self, xy1, xy2):
        """
        Find the indicies of xy1 that overlap with xy2, where both are
        n x 2 dimensional arrays of (x,y) pixels.
        """
        
        assert xy1.shape[1] == 2
        assert xy2.shape[1] == 2
        
        p_ind_x = np.intersect1d( np.where( xy1[:,0] > xy2[:,0].min() )[0], 
                                  np.where( xy1[:,0] < xy2[:,0].max() )[0] )
        p_ind_y = np.intersect1d( np.where( xy1[:,1] > xy2[:,1].min() )[0], 
                                  np.where( xy1[:,1] < xy2[:,1].max() )[0] )                                    
        p_inds = np.intersect1d(p_ind_x, p_ind_y)
        
        return p_inds
        
        
    def _overlap_implicit(self, xy, grid):
        """
        xy   : ndarray, float, 2D
        grid : (basis, size, corner)
        """
        assert xy.shape[1] == 2
        
        # grid:   corner       basis           size
        min_x = grid[2][0]
        max_x = grid[2][0] + grid[0][0] * (grid[1][0] - 1)
        min_y = grid[2][1]
        max_y = grid[2][1] + grid[0][1] * (grid[1][1] - 1)
        
        p_ind_x = np.intersect1d( np.where( xy[:,0] > min_x )[0], 
                                  np.where( xy[:,0] < max_x )[0] )
        p_ind_y = np.intersect1d( np.where( xy[:,1] > min_y )[0], 
                                  np.where( xy[:,1] < max_y )[0] )                                    
        p_inds = np.intersect1d(p_ind_x, p_ind_y)
        
        return p_inds
        
        
    def _implicit_interpolation(self):
        """
        Interpolate onto a polar grid from an `implicit` detector geometry.
        
        This detector geometry is specified by the x and y pixel spacing,
        the number of pixels in the x/y direction, and the top-left corner
        position.
        
        NOTE: The interpolation is performed in cartesian real (xyz) space.
        """
                
        # loop over all the arrays that comprise the detector and use
        # grid interpolation on each
        
        int_start = 0 # start of intensity array correpsonding to `grid`
        int_end   = 0 # end of intensity array correpsonding to `grid`
        
        # convert the polar to cartesian coords for comparison to detector
        pgr = self.polar_grid_as_real_cart
        
        for k,grid in enumerate(self.detector._grid_list):
            
            basis, size, corner = grid
            n_int = int(size[0] * size[1])
            int_end += n_int
        
            if size[2] != 1:
                logger.debug('Warning: Z dimension not flat')
        
            if (basis[1] == 0.0) or (basis[0] == 0.0):
                raise RunTimeError('Implicit detector is flat in either x or y!'
                                   ' Cannot be interpolated.')
            
            i = self.intensities[int_start:int_end]
            int_start += n_int
            
            # find the indices of the polar grid pixels that are inside the
            # boundaries of the current detector array we're interpolating over
            p_inds = self._overlap_implicit(pgr, grid)
            
            if len(p_inds) == 0:
                logger.warning('Detector array (%d/%d) had no pixels inside the \
                interpolation area!' % (k+1, len(self.detector.grid_list)) )
                continue
                        
            # interpolate onto the polar grid & update the inverse mask
            # perform the interpolation. 
            interp = Bcinterp(i, basis[0], basis[1], size[0], size[1], corner[0], corner[1])
            
            self.polar_intensities[p_inds] = interp.evaluate(pgr[p_inds,0], pgr[p_inds,1])
            self.polar_mask[p_inds] = np.bool(False)
            
        self._mask()
        
        return
        
        
    def _unstructured_interpolation(self):
        """
        Perform an interpolation to polar coordinates assuming that the detector
        pixels do not form a rectangular grid.
        
        NOTE: The interpolation is performed in polar momentum (q) space.
        """
        
        # interpolate onto polar grid : delauny triangulation + linear
        xy = np.zeros(( self.detector.polar.shape[0], 2 ))
        xy[:,0] = self.detector.k * np.sqrt( 2.0 - 2.0 * np.cos(self.detector.polar[:,1]) ) # |q|
        xy[:,1] = self.detector.polar[:,2] # phi
        
        # because we're using a "square" interplation method, wrap around one
        # set of polar coordinates to capture the periodic nature of polar coords
        add = np.where( xy[:,1] == xy[:,1].min() )[0]
        xy_add = xy[add]
        xy_add[:,1] += 2.0 * np.pi
        aug_xy  = np.concatenate(( xy, xy_add ))
        aug_int = np.concatenate(( self.intensities, self.intensities[add] ))
        
        # do the interpolation
        z_interp = interpolate.griddata( aug_xy, aug_int, self.polar_grid,
                                         method='linear', fill_value=np.nan )

        self.polar_intensities = z_interp.flatten()

        # mask missing pixels (outside convex hull)
        nan_ind = np.where( np.isnan(z_interp) )[0]
        self.polar_intensities[nan_ind] = 0.0
        not_nan_ind = np.delete(np.arange(z_interp.shape[0]), nan_ind)
        
        self.polar_mask[not_nan_ind] = np.bool(False)
        self._mask()
        
        return
        
        
    def _mask(self):
        """
        Apply the mask in self.polar_mask to the masked array 
        self.polar_intensities.
        """
        self.polar_intensities = np.ma.array(np.array(self.polar_intensities), 
                                             mask=self.polar_mask) 
        
        
    def _unmask(self):
        """
        Reveals all masked coordinates.
        """
        self.polar_intensities = np.array(self.polar_intensities)
        
        
    def _apply_mask(self):
        """
        Sets the polar mask to mask the pixels indicated by the mask argument.
        """
        
        # THIS FUNCTION WILL CHANGE TO ACCOMODATE MORE FLEXIBLE MASKING
        
        # if we have no work to do, let's escape!
        if np.sum(self.real_mask) == 0:
            return
        
        # perform a fixed-radius nearest-neighbor search, masking all the pixels
        # on the polar grid that are within one-half pixel spacing from a masked
        # pixel on the cartesian grid
        
        xyz = self.detector.reciprocal
        pgc = self.polar_grid_as_cart
        
        # the max distance, r^2 - a factor of 10 helps get the spacing right
        r2 = np.sum( np.power( xyz[0] - xyz[1], 2 ) ) * 10.0
        masked_cart_points = xyz[self.real_mask,:2]
        
        # todo : currently slow, can be done faster? Could Weave + OMP
                
        for mp in masked_cart_points:
            d2 = np.sum( np.power( mp - pgc[:,:2], 2 ), axis=1 )
            self.polar_mask[ d2 < r2 ] = np.bool(True)
        
        self._mask()
        
        return
        
        
    def _nearest_q(self, q):
        """
        Get the value of q nearest to the argument that is on our computed grid.
        """
        if (q % self.q_spacing == 0.0) and (q > self.q_min) and (q < self.q_max):
            pass
        else:
            q = self.q_values[ bisect_left(self.q_values, q, hi=len(self.q_values)-1) ]
            logger.debug('Passed value `q` not on grid -- using closest '
                         'possible value')
        return q
        
        
    def _nearest_phi(self, phi):
        """
        Get value of phi nearest to the argument that is on our computed grid.
        """
        # phase shift
        phi = phi % (2*np.pi)
        
        if (phi % self.phi_spacing == 0.0):
            pass
        else:
            phi = self.phi_values[ bisect_left(self.phi_values, phi, hi=self.num_phi-1) ]
            logger.debug('Passed value `phi` not on grid -- using closest '
                         'possible value')
        return phi
        
        
    def _nearest_delta(self, delta):
        """
        Get the value of delta nearest to a multiple of phi_spacing.
        
        This is really just the same as _nearest_phi() right now (BAD)
        """
        # phase shift
        delta = delta % (2*np.pi)
        
        if (delta % self.phi_spacing == 0.0):
            pass
        else:
            delta = self.phi_values[ bisect_left(self.phi_values, delta, hi=self.num_phi-1) ]
            logger.debug('Passed value `delta` not on grid -- using closest '
                         'possible value')
        return delta


    def _q_index(self, q):
        """
        Quick return of all self.polar_intensities with a specific `q`
        """
        # recall : q values are tiled
        q = self._nearest_q(q)
        start = int(q/self.q_spacing) - int(self.q_min/self.q_spacing)
        inds = np.arange(start, self.num_datapoints, self.num_q)
        return inds
        
        
    def _phi_index(self, phi):
        """
        Quick return of all self.polar_intensities with a specific `phi`
        """
        # recall : phi values are repeated
        phi = self._nearest_phi(phi)
        start = int(phi/self.phi_spacing) * self.num_q
        inds = np.arange(start, start+self.num_q)
        return inds

        
    def _intensity_index(self, q, phi):
        """
        Returns the index of self.polar_intensities that matches the passed values
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
            The index of self.polar_intensities that is closest to the passed values,
            such that self.polar_intensities[index] -> I(q,phi)
        """
        
        q = self._nearest_q(q)
        phi = self._nearest_phi(phi)
        
        qpos = int(q/self.q_spacing) - int(self.q_min/self.q_spacing)
        ppos = int(phi/self.phi_spacing) * self.num_q
        index = qpos + ppos
        
        return index
        

    def I(self, q, phi):
        """
        Return the intensity a (q,phi).
        """
        return np.array(self.polar_intensities)[self._intensity_index(q,phi)]
    
        
    def I_ring(self, q):
        """
        Return the intensities around a ring at `q`
        """
        q = self._nearest_q(q)
        ind = self._q_index(q)
        return np.array(self.polar_intensities)[ind]
    
        
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
        ind = self._q_index(q)
        
        # the two-step type conversion below ensures that (1) masked values
        # aren't included in the calc, and (2) if all the values are masked
        # we get intensity = 0.0
        intensity = float( np.array( (self.polar_intensities[ind]).mean() ))
        
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
            intensity_profile[i,1] = np.float(self.qintensity(q))

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
        a = smooth(intensity[:,1], beta=smooth_strength)
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
        
        q1 = self._nearest_q(q1) 
        q2 = self._nearest_q(q2)
        delta = self._nearest_phi(delta)
        
        i = int(delta / self.phi_spacing)
        logger.debug('Correlating %d indicies away' % i)
        
        x = self.I_ring(q1)
        y = self.I_ring(q2)
        y = np.roll(y, i)
        
        x -= x.mean()
        y -= y.mean()
        
        # this should work with masking
        corr = np.mean(x * y) / (x.std() * y.std())
        
        return corr
        
        
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

        # As DTM has pointed out, here we need to be a little careful with
        # how we deal with masked pixels. If we subtract the means from the 
        # pixels first, then perform the correlation, we can safely ignore
        # masked pixels so long as (1) we make sure their value is zero, and
        # (2) we don't include them in the count `n_delta` of total corration
        # pairs.
        
        q1 = self._nearest_q(q1)
        q2 = self._nearest_q(q2)
        
        x = self.I_ring(q1)
        y = self.I_ring(q2)
        assert len(x) == len(y)
        
        logger.debug("Correlating rings at %f / %f" % (q1, q2))
        
        x -= x.mean()
        y -= y.mean()
        
        # plug in a zero at all mask positions
        if np.ma.isMaskedArray(x):
            x[x.mask] = 0.0
        if np.ma.isMaskedArray(y):
            y[y.mask] = 0.0
                
        correlation_ring = np.zeros((len(x), 2))
        correlation_ring[:,0] = self.phi_values
        
        # use d-FFT + convolution thm
        ffx = fftpack.fft(x)
        ffy = fftpack.fft(y)
        iff = np.real(fftpack.ifft( ffx * np.conjugate(ffy) ))
        
        correlation_ring[:,1] = iff / (np.linalg.norm(x) * np.linalg.norm(y))
        
        return correlation_ring
        
        
    def correlate_all_rings(self):
        """
        Automatically detects the position of scattering rings and performs the
        correlation of all pairs of those rings.

        Returns
        -------
        all_rings : odin.xray.CorrelationCollection
            A CorrelationCollection object containing data for all combinations
            of correlated rings.
        """
        return CorrelationCollection(self)
        
    @classmethod
    def simulate(cls, traj, num_molecules, detector, traj_weights=None, 
                 force_no_gpu=False, device_id=0):
        """
        Simulate a scattering 'shot', i.e. one exposure of x-rays to a sample, and
        return that as a Shot object (factory function).

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
    
        detector : odin.xray.Detector
            A detector object the shot will be projected onto.
    
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
            The index of the GPU to run on.
    
        Returns
        -------
        shot : odin.xray.Shot
            A shot instance, containing the simulated shot.
        """
        I = simulate_shot(traj, num_molecules, detector, traj_weights=traj_weights, 
                          force_no_gpu=force_no_gpu, device_id=device_id)
        shot = Shot(I, detector)
        return shot
        
        
    def _unpack_interp_params(self, interp_params):
        """ Extract/inject saved parameters specifying the polar interpolation """
        self.phi_spacing = interp_params[0]
        self.q_min       = interp_params[1]
        self.q_max       = interp_params[2]
        self.q_spacing   = interp_params[3]
        return


    def _pack_interp_params(self):
        """ Package the interpolation parameters for saving """
        interp_params = [self.phi_spacing, self.q_min, self.q_max, self.q_spacing]
        return np.array(interp_params)
        
        
    # So as to not duplicate code/logic, I am going to employ a wrapped version
    # of the Shotset save/load methods here...
        
    def save(self, filename, save_interpolation=True):
        """
        Writes the current Shot data to disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file to save.
        """
        ss = Shotset([self])
        ss.save(filename)


    @classmethod
    def load(cls, filename):
        """
        Loads the a Shot from disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file.

        Returns
        -------
        shotset : odin.xray.Shotset
            A shotset object
        """
        ss = Shotset.load(filename)
        
        if len(ss) != 1:
            logger.warning('You loaded a .shot file that contains multiple shots'
                           ' into a single Shot instance... taking only the'
                           ' first shot of the set (look into Shotset.load()).')
                           
        return ss[0]


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
        
        
    def __len__(self):
        return len(self.shots)
        
        
    def __iter__(self):
        for i in xrange(self.num_shots):
            yield self.shots[i]
        
            
    def __getitem__(self, key):
        return self.shots[key]
        
        
    def __add__(self, other):
        if not isinstance(other, Shotset):
            raise ValueError('Cannot add types: %s and Shotset' % type(other))
        return Shotset( self.shots.extend(other.shots) )
        
        
    def _check_qvectors_same(self, epsilon=1e-6):
        """
        For a lot of the below to work, we need to make sure that all of the 
        q-phi grids in the shots are the same. If some are different, here
        we recalculate them.
        """
        q_phis = self.shots[0].polar_grid
        for shot in self.shots:
            diff = np.sum(np.abs(shot.polar_grid - q_phis))
            if diff > epsilon:
                raise ValueError('Detector values in Shotset not consistent '
                                  ' across set. Homogenize interpolated polar'
                                  ' grid using Shot.interpolate_to_polar()')
    
    
    def I(self, q, phi):
        """
        Return the intensity at (q,phi).
        """
        
        intensity = 0.0
    
        for shot in self.shots:
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
    
        for shot in self.shots:
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
        
        intensity_profile = self.shots[0].intensity_profile()
    
        for shot in self.shots[1:]:
            intensity_profile[:,1] += shot.intensity_profile()[:,1]
            
        intensity_profile[:,1] /= float(self.num_shots)
        
        return intensity_profile


    def intra(self,q1,q2):
    	"""
    	computes intra-shot correlation ffts
    	"""
    	shots = self.shots
    	n_phi = len(shots[0].phi_values)
    	intraCors = [np.abs(np.fft.fft (s.correlate_ring_brute(q1,q2)[:,1], n_phi)) for s in shots]
    	return intraCors


    def inter(self,q1,q2,n_inter=0):
       	"""
    	computes inter-shot correlation ffts
	
    	Paramters
    	---------
    	q1 : float
    	    magnitude of first position to correlate
	    
    	q2 : float
    	    magnitude of second position to correlate
	    
    	n_inter : int , optional
    	    number of inter-shot correlation ffts to compute
	
    	Returns
    	--------
    	list of np.ndarrays
	
    	"""
    	shots = self.shots
    	n_shots = len(shots)
	
    	if n_shots == 1:
    		raise ValueError("Cannot compute inter shot correlations with 1 shot")
		
        # I arbitrarily picked 0.6
        # TJL to DERMEN: what is this and do we need it?
    	if n_inter > 0.6 * (n_shots+1)*n_shots/2 :
    		print "Might take a long time to find",n_inter,"unique inter-shot pairs from",n_shots
    		print "shots. Please choose n_inter <",int(0.6 *  (n_shots+1)*n_shots/2 ),"."
    		raise RuntimeError()
    		
	
    	if n_inter==0:
    	    n_inter=n_shots
	    	
    	interCors = []
    	for s1,s2 in rand_pairs(n_shots,n_inter):
    	    shot1 = shots[s1]
    	    shot2 = shots[s2]
    	    I1 = shot1.I_ring(q1)
    	    I2 = shot2.I_ring(q2)

            # note: when the args passed to correlate_ring_brute are numpy.ndarrays
            # the only call to self is self.phi_values, which should be the same for
            # shot1 or shot2
    	    c12= shot1.correlate_ring_brute(I1,I2)[:,1]
    	    interCors.append( np.abs( np.fft.fft(c12,len(c12) ) ))
	
    	return interCors
        
        
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
    
        for shot in self.shots:
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
        correlation_ring = self.shots[0].correlate_ring(q1, q2)
    
        for shot in self.shots[1:]:
            correlation_ring[:,1] += shot.correlate_ring(q1, q2)[:,1]
            
        correlation_ring[:,1] /= float(self.num_shots)
        
        return correlation_ring
        
            
    @classmethod
    def simulate(cls, traj, num_molecules, detector, num_shots,
                 traj_weights=None, force_no_gpu=False, device_id=0):
        """
        Simulate many scattering 'shot's, i.e. one exposure of x-rays to a sample, and
        return that as a Shot object (factory function).

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

        detector : odin.xray.Detector
            A detector object the shot will be projected onto.

        num_molecules : int
            The number of molecules estimated to be in the `beam`'s focus.
        
        num_shots : int
            The number of shots to perform and include in the Shotset.

        traj_weights : ndarray, float
            If `traj` contains many structures, an array that provides the Boltzmann
            weight of each structure. Default: if traj_weights == None, weights
            each structure equally.

        force_no_gpu : bool
            Run the (slow) CPU version of this function.

        Returns
        -------
        shotset : odin.xray.Shotset
            A Shotset instance, containing the simulated shots.
        """
        
        device_id = int(device_id)
    
        shotlist = []
        for i in range(num_shots):
            I = simulate_shot(traj, num_molecules, detector, traj_weights=traj_weights, 
                              force_no_gpu=force_no_gpu, device_id=device_id)
            shot = Shot(I, detector)
            shotlist.append(shot)
            
            logger.info('Finished shot %d/%d on device %d' % (i+1, num_shots, device_id) )
        
        return Shotset(shotlist)
        
                
    def save(self, filename, save_interpolation=True):
        """
        Writes the current Shotset data to disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file to save.
            
        Optional Parameters
        -------------------
        save_interpolation : bool
            Save the polar interpolation along with the cartesian intensities.
        """

        if not filename.endswith('.shot'):
            filename += '.shot'
            
        # in the below, I use np.array() to unmask arrays before saving

        shotdata = {}
        for i in range(self.num_shots):
            shotdata[('shot%d' % i)] = np.array(self.shots[i].intensities)
            shotdata[('shot%d_mask' % i)] = self.shots[i].real_mask
            if save_interpolation:
                shotdata[('shot%d_polar_intensities' % i)] = np.array(self.shots[i].polar_intensities)
                shotdata[('shot%d_polar_mask' % i)]        = self.shots[i].polar_mask
                shotdata[('shot%d_interp_params' % i)]     = self.shots[i]._pack_interp_params()

        io.saveh(filename, 
                 num_shots = np.array([self.num_shots]),
                 detector  = self.shots[0].detector._to_serial(),
                 **shotdata)

        logger.info('Wrote %s to disk.' % filename)


    @classmethod
    def load(cls, filename, to_load=None):
        """
        Loads the a Shotset from disk. Must be `.shot` or `.cxi` format.

        Parameters
        ----------
        filename : str
            The path to the shotset file.
            
        to_load : ndarray/list, ints
            The indices of the shots in `filename` to load. Can be used to sub-
            sample the shotset.

        Returns
        -------
        shotset : odin.xray.Shotset
            A shotset object
        """

        if filename.endswith('.shot'):
            hdf = io.loadh(filename)

            num_shots   = int(hdf['num_shots'])
            
            # figure out which shots to load
            if to_load == None:
                to_load = range(num_shots)
            else:
                try:
                    to_load = np.array(to_load)
                except:
                    raise ValueError('`to_load` must be a ndarry/list of ints')

            # going to maintain backwards compatability -- this should be
            # deprecated ASAP though....
            try:
                d = Detector._from_serial(hdf['detector']) # new (keep)
            except KeyError as e:
                # old
                logger.warning('WARNING: Loaded deprecated Shotset... please re- '
                                'save this Shotset using Shotset.save() '
                                'and use the newly saved version! This '
                                'will automatically upgrade your data to the '
                                'latest version.')
                num_shots = hdf['num_shots'][0]
                xyz = hdf['dxyz']
                path_length = hdf['dpath_length'][0]
                k = hdf['dk'][0]
                d = Detector(xyz, path_length, k)
        
            list_of_shots = []
            for i in to_load:
                i = int(i)
                intensities = hdf[('shot%d' % i)]
                
                # some older files may not have any mask, so be gentle
                if ('shot%d_mask' % i) in hdf.keys():
                    real_mask = hdf[('shot%d_mask' % i)]
                else:
                    real_mask = None
                
                # load saved polar interp if it exists
                if ('shot%d_polar_intensities' % i) in hdf.keys():
                    polar_intensities = hdf[('shot%d_polar_intensities' % i)]
                    polar_mask = hdf[('shot%d_polar_mask' % i)]
                    interp_params = hdf[('shot%d_interp_params' % i)]
                    iv = (polar_intensities, polar_mask, interp_params)
                else:
                    iv = None
                
                s = Shot(intensities, d, mask=real_mask, interpolated_values=iv)
                list_of_shots.append(s)
            
        elif filename.endswith('.cxi'):
            raise NotImplementedError() # todo
            
        else:
            raise ValueError('Must load a shotset file [.shot, .cxi]')

        hdf.close

        return Shotset(list_of_shots)


class CorrelationCollection(object):
    """
    A class to manage a large set of ring correlations, likely computed from a
    Shot or Shotset.
    
    The central data structure here is a dictionary self._correlation_data
    that contains the correlation rings. This dict is keyed by the q-values 
    for which each ring was computed, e.g.
    
        self._correlation_data[(q1,q2)] -> correlation_ring
        
    note len(correlation_ring) = len(phi_values), and it *must* be that
    q1 >= q2.
    """
    
    def __init__(self, shot, q_values=None):
        """
        Generate a CorrelationCollection from a shot or shotset class
        
        Parameters
        ----------
        shot : odin.xray.Shot OR odin.xray.shotset
            The shot/shotset object from which to generate the 
            CorrelationCollection.
            
        Optional Parameters
        -------------------
        q_values : ndarray OR list OF floats
            The specific q-values (in inv. angstroms) at which to correlate. If
            `None`, then will automatically locate Bragg rings and correlate
            all combinations of those rings
        """
        
        self._correlation_data = {}
        
        if q_values == None:
            self.q_values = shot.q_values[shot.intensity_maxima()]
        else:
            self.q_values = q_values
            
        self.phi_values = shot.phi_values
        self.k = shot.detector.k
        
        self.num_q   = len(self.q_values)
        self.num_phi = len(self.phi_values)
        
        for i in range(self.num_q):
            for j in range(i, self.num_q):
                q1 = self.q_values[i]
                q2 = self.q_values[j]
                
                if not hasattr(self, 'deltas'):
                    self.deltas = shot.correlate_ring(q1, q2)[:,0]
                else:
                    assert np.all(self.deltas == shot.correlate_ring(q1, q2)[:,0])
                    
                self._correlation_data[(q1,q2)] = shot.correlate_ring(q1, q2)[:,1]
                
            
    def ring(self, q1, q2):
        """
        Parameters
        ----------
        q1, q2 : float
            The q-values (in inv. angstroms) for which to get the ring.
        """
        if q1 < q2:
            return self._correlation_data[(q2,q1)]
        else:
            return self._correlation_data[(q1,q2)]
            
            
    def legendre_coeffecients(self, order=20, report_error=False):
        """
        Project the correlation functions onto a set of legendre polynomials,
        and return the coefficients of that projection.
        
        Optional Parameters
        -------------------
        order : int
            The order at which to truncate the polynomial expansion. If none,
            will use epsilon parameter to decide order.
        """
        
        # initialize space for coefficients        
        Cl = np.zeros((order, self.num_q, self.num_q))
            
        for i in range(self.num_q):
            for j in range(i,self.num_q):
                
                # compute psi, the angle between the scattering vectors
                t1 = np.arctan( self.q_values[i] / (2.0*self.k) ) # theta1
                t2 = np.arctan( self.q_values[j] / (2.0*self.k) ) # theta2

                psi = np.arccos( np.cos(t1)*np.cos(t2) + np.sin(t1)*np.sin(t2) \
                                 * np.cos( self.deltas * 2. * np.pi/float(self.num_phi) ) )
                                                     
                q1 = self.q_values[i]
                q2 = self.q_values[j]
                
                x = order*2 - 1
                c, fit_data = np.polynomial.legendre.legfit(np.cos(psi), 
                                  self._correlation_data[(q1,q2)], x, full=True)
                                  
                assert len(c) == order * 2
                c = c[::2]     # discard odd values
                assert len(c) == Cl.shape[0]
                
                c /= c.sum()   # normalize
                Cl[:,i,j] = c
                Cl[:,j,i] = c  # copy it to the lower triangle too
        
        if report_error:
            d = 0.0
            for i in range(self.num_q):
                for j in range(i,self.num_q):
                    
                    c = np.zeros( 2 * Cl.shape[0] )
                    c[::2] = Cl[:,i,j]
                    pred = np.polynomial.legendre.legval( np.cos(psi), c)
                    
                    q1 = self.q_values[i]
                    q2 = self.q_values[j]
                    actu = self._correlation_data[(q1,q2)]

                    d += np.sum(np.abs(pred - actu))
                
            logger.info('Expansion error: %f' % (d,))    
                    
        return Cl
        
    
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
    
    # NOTES ON DATA TYPES
    # all arrays should be float32 / int32
    # output array gets upcast to float64 before being returned
    # native python ints are expected by swig - not stand alone np.int32's
    
    logger.debug('Performing scattering simulation...')
    logger.debug('Simulating %d copies in the dilute limit' % num_molecules)

    # stupidity check
    if type(device_id) != int:
        raise ValueError('device_id must be type int')

    if traj_weights == None:
        traj_weights = np.ones( traj.n_frames )
    traj_weights /= traj_weights.sum()
        
    num_per_shapshot = np.random.multinomial(num_molecules, traj_weights)

    # get detector vectors
    
    if isinstance(detector, Detector):    
        qx = detector.reciprocal[:,0].astype(np.float32)
        qy = detector.reciprocal[:,1].astype(np.float32)
        qz = detector.reciprocal[:,2].astype(np.float32)
        num_q = len(qx)
        assert detector.num_pixels == num_q
    elif isinstance(detector, np.ndarray):
        qx = detector[:,0].astype(np.float32)
        qy = detector[:,1].astype(np.float32)
        qz = detector[:,2].astype(np.float32)
        num_q = len(qx)
    else:
        raise ValueError('`detector` must be {odin.xray.Detector, np.ndarray}')

    # get cromer-mann parameters for each atom type
    # renumber the atom types 0, 1, 2, ... to point to their CM params
    aZ = np.array([ a.element.atomic_number for a in traj.topology.atoms() ])
    atom_types = np.unique(aZ)
    num_atom_types = len(atom_types)

    # if `num_atom_types` > 10, we're in trouble
    if num_atom_types > 10:
        raise Exception('Fatal Error. Your molecule has >10 unique atom types '
                        'but the GPU code cannot handle more than 10 due to '
                        'code requirements. You can recompile the GPU kernel '
                        'to fix this -- see file odin/src/cuda/gpuscatter.cu')

    cromermann = np.zeros(9*num_atom_types, dtype=np.float32)
    aid = np.zeros( len(aZ), dtype=np.int32 )

    for i,a in enumerate(atom_types):
        ind = i * 9
        try:
            cromermann[ind:ind+9] = np.array(cromer_mann_params[(a,0)]).astype(np.float32)
        except KeyError as e:
            logger.critical('Element number %d not in Cromer-Mann form factor parameter database' % a)
            raise ValueError('Could not get critical parameters for computation')
        aid[ aZ == a ] = np.int32(i)
        
    # do the simulation, scan over confs., store in `intensities`
    #intensities = mp.Array( 'f', [0.0]*num_q )
    intensities = np.zeros(num_q, dtype=np.float32)    

    for i,num in enumerate(num_per_shapshot):
        num = int(num)
        if num > 0: # else, we can just skip...
        
            # pull xyz coords
            xyz = traj.xyz[i,:,:] * 10.0 # convert nm -> ang.
            rx = xyz[:,0].flatten().astype(np.float32)
            ry = xyz[:,1].flatten().astype(np.float32)
            rz = xyz[:,2].flatten().astype(np.float32)

            # choose the number of molecules & divide work between CPU & GPU
            # GPU is fast but can only do multiples of 512 molecules - run
            # the remainder on the CPU
            if force_no_gpu or (not GPU):
                num_cpu = num
                num_gpu = 0
                bpg = 0
                logger.debug('Running CPU-only computation')
            else:
                num_cpu = num % 512
                num_gpu = num - num_cpu
                bpg = num_gpu / 512 # this is 512 * the number we'll run on the GPU
            
            logger.info('Running %d molecules from snapshot %d...' % (num, i))  

            # multiprocessing cannot return values, so generate a helper function
            # that will dump returned values into a shared array
            threads = []
            def multi_helper(name, fargs):
                """ a helper function that performs either CPU or GPU calcs """                
                if name == 'cpu':
                    function = cpuscatter.CPUScatter
                elif name == 'gpu':
                    function = gpuscatter.GPUScatter
                else:
                    raise RuntimeError('cpu/gpu are the only names allowed, got: %s' % name)
                
                result = function(*fargs)
                intensities[:] += result.this[1]
                return

            # run dat shit
            if num_cpu > 0:
                logger.debug('Running CPU scattering code (%d/%d)...' % (num_cpu, num))
                # generate random numbers for the rotations in python (much easier)
                rand1 = np.random.rand(num_cpu).astype(np.float32)
                rand2 = np.random.rand(num_cpu).astype(np.float32)
                rand3 = np.random.rand(num_cpu).astype(np.float32)
                cpu_args = (num_cpu, qx, qy, qz, rx, ry, rz, aid,
                            cromermann, rand1, rand2, rand3, num_q)
                t_cpu = Thread(target=multi_helper, args=('cpu', cpu_args))
                t_cpu.start()
                threads.append(t_cpu)                

            if bpg > 0:
                logger.debug('Sending calculation to GPU device...')
                # generate random numbers for the rotations in python (much easier)
                rand1 = np.random.rand(num_gpu).astype(np.float32)
                rand2 = np.random.rand(num_gpu).astype(np.float32)
                rand3 = np.random.rand(num_gpu).astype(np.float32)
                gpu_args = (device_id, bpg, qx, qy, qz, rx, ry, rz, aid,
                            cromermann, rand1, rand2, rand3, num_q)
                t_gpu = Thread(target=multi_helper, args=('gpu', gpu_args))
                t_gpu.start()
                threads.append(t_gpu)
                
            # ensure child processes have finished
            for t in threads:
                t.join()

    # return data in most expected format
    intensities = intensities.astype(np.float64)

    # normalize intensities
    intensities /= intensities.sum()
        
    # check for NaNs in output
    if np.isnan(np.sum(intensities)):
        raise RuntimeError('Fatal error, NaNs detected in scattering output!')
    
    # check for negative values in output
    if len(intensities[intensities < 0.0]) != 0:
        raise RuntimeError('Fatal error, negative intensities detected in scattering output!')    

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
        
    for i in range(4):
        fi  = cromermann[8]
        fi += cromermann[i] * np.exp( cromermann[i+5] * qo)
        
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



