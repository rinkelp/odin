# THIS FILE IS PART OF ODIN


"""
Classes, methods, functions for use with xray scattering experiments.
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
#logger.setLevel('DEBUG')

import cPickle
from bisect import bisect_left

import numpy as np
from scipy import interpolate, fftpack
from scipy.ndimage import filters
from scipy.special import legendre

from odin.math2 import arctan3, rand_pairs, smooth
from odin import scatter
from odin.interp import Bcinterp

from mdtraj import trajectory, io

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
    
    def __init__(self, photons_scattered_per_shot, **kwargs):
        """
        Generate an instance of the Beam class.
        
        Parameters
        ----------
        photons_scattered_per_shot : int
            The average number of photons scattered per shot.
        
        **kwargs : dict
            Exactly one of the following, in the indicated units
            -- energy:     keV
            -- wavelength: angstroms
            -- frequency:  Hz
            -- wavenumber: inverse angstroms
        """
        
        self.photons_scattered_per_shot = photons_scattered_per_shot
        
        # make sure we have only one argument
        if len(kwargs) != 1:
            raise KeyError('Expected exactly one argument, got %d' % (len(args)+1) )
        
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


class BasisGrid(object):
    """
    A class representing a set of rectangular grids in space -- specifically,
    x-ray scattering detectors. Does not contain all the metadata associated
    with a full-fledged Detector class (e.g. the wavelength, etc).

    Note that the geometry below is definied in "slow" and "fast" scan 
    dimensions. These are simply the two dimensions that define the plane
    a single rectangular pixel grid lives in. They may also be called the x and
    y dimensions without any loss of generality.

    Note on units: units are arbitrary -- all the units must be the same for
    this to work. We don't keep track of units here.

    The class is a set of rectangular grids, with each grid defined by four
    quantities:

        -- p vector : DEFINES A GRIDS POSITION IN SPACE.
                      The vector between a chosen origin (possibly interaction 
                      site) and the corner of the grid that is smallest in both 
                      slow and fast (x/y) dimensions of the coordinate system. 
                      Usually this will be the "bottom left" corner, but due to 
                      the generality of the coordinates used, this is not 
                      necessarily true.

        -- s/f vect : DEFINES A GRIDS ORIENTATION IN SPACE
                      Vectors pointing along the slow/fast-scan direction,
                      respectively. These define the plane containing the pixels.
                      The magnitudes of these vectors defines the size of the
                      pixel in that dimension.

        -- shape    : DEFINES GRID DIMENSIONS
                      The number of pixels in the fast/slow direction. Ints.
    """


    def __init__(self, list_of_grids=[]):
        """
        Initialize a BasisGrid object.

        Parameters
        ----------
        list_of_grids : list
            A list of tuples of the form  (p, s, f, shape). See the doc
            for the `add_grid` method on this class for more information. May
            be an empty list (default) in which case a GridList with no pixels
            is created.

        See Also
        --------
        add_grid
        add_grid_using_center
        """

        if not type(list_of_grids) == list:
            raise TypeError('`list_of_grids` must be a list')

        self.num_grids  = 0
        self._ps        = [] # p-vectors
        self._ss        = [] # slow-scan vectors
        self._fs        = [] # fast-scan vectors
        self._shapes    = [] # shapes

        if len(list_of_grids) > 0:
            for grid in list_of_grids:
                self.add_grid(*grid)

        return


    def _check_valid_basis(self, p, s, f, shape):
        """
        Check to make sure that all the inputs look good.
        """

        if not (p.shape == (3,)) and (s.shape == (3,)) and (f.shape == (3,)):
            raise ValueError('`p`, `s`, `f` must be 3-vectors')

        if not (len(shape) == 2):
            raise ValueError('`shape` must be len 2')

        return


    def _assert_list_sizes(self):
        """
        A simple sanity check
        """
        assert len(self._ps)     == self.num_grids
        assert len(self._ss)     == self.num_grids
        assert len(self._fs)     == self.num_grids
        assert len(self._shapes) == self.num_grids
        return


    def add_grid(self, p, s, f, shape):
        """
        Add a grid (detector array) to the basis representation.

        Parameters
        ----------
        p : np.ndarray, float
            3-vector from the origin to the pixel on the grid with 
            smallest coordinate in all dimensions.

        s : np.ndarray, float
            3-vector pointing in the slow scan direction

        f : np.ndarray, float
            3-vector pointing in the slow scan direction    

        shape : tuple or list of float
            The number of pixels in the (slow, fast) directions. Len 2.

        See Also
        --------    
        add_grid_using_center
        """
        self._check_valid_basis(p, s, f, shape)
        self._ps.append(p)
        self._ss.append(s)
        self._fs.append(f)
        self._shapes.append(shape)
        self.num_grids += 1
        self._assert_list_sizes()
        return


    def add_grid_using_center(self, p_center, s, f, shape):
        """
        Add a grid (detector array) to the basis representation. Here, though,
        the p-vector points to the center of the array instead of the slow/fast
        smallest corner.

        Parameters
        ----------
        p_center : np.ndarray, float
            3-vector from the origin to the center of the grid.

        s : np.ndarray, float
            3-vector pointing in the slow scan direction

        f : np.ndarray, float
            3-vector pointing in the slow scan direction

        shape : tuple or list of float
            The number of pixels in the (slow, fast) directions. Len 2.
        """

        p_center = np.array(p_center)
        if not p_center.shape == (3,):
            raise ValueError('`p_center` must have shape (3,)')

        # just compute where `p` is then add the grid as usual
        x = (np.array(shape) - 1)
        center_correction =  ((x[0] * s) + (x[1] * f)) / 2.
        p  = p_center.copy()
        p -= center_correction

        self.add_grid(p, s, f, shape)

        return


    def get_grid(self, grid_number):
        """
        Return a grid for grid `grid_number`.

        Parameters
        ----------
        grid_number : int
            The index of the grid to get.

        Returns
        -------
        p_center : np.ndarray, float
            3-vector from the origin to the center of the grid.

        s : np.ndarray, float
            3-vector pointing in the slow scan direction

        f : np.ndarray, float
            3-vector pointing in the slow scan direction

        shape : tuple or list of float
            The number of pixels in the (slow, fast) directions. Len 2.
        """

        if grid_number >= self.num_grids:
            raise ValueError('Only %d grids in object, you asked for the %d-th'
                             ' (zero indexed)' % (self.num_grids, grid_number))

        grid_tuple = (self._ps[grid_number], self._ss[grid_number], 
                      self._fs[grid_number], self._shapes[grid_number])

        return grid_tuple


    def to_explicit(self):
        """
        Return the entire grid as an n x 3 array, defining the x,y,z positions
        of each pixel.

        Returns
        -------
        xyz : np.ndarray, float
            An N x 3 array of the x,y,z positions of each pixel. Note that this
            is a flattened version of what you get for each grid individually
            using `grid_as_explicit`.

        See Also
        --------
        grid_as_explicit
        """
        ex_grids = [ self.grid_as_explicit(i) for i in range(self.num_grids) ]
        xyz = np.concatenate([ g.reshape((g.shape[0]* g.shape[1], 3)) for g in ex_grids ])
        return xyz


    def grid_as_explicit(self, grid_number):
        """
        Get the x,y,z coordiantes for a single grid.

        Parameters
        ----------
        grid_number : int
            The index of the grid to get.

        Returns
        -------
        xyz : np.ndarray, float
            An (shape) x 3 array of the x,y,z positions of each pixel

        See Also
        --------
        to_explicit
        """

        p, s, f, shape = self.get_grid(grid_number)

        # xyz = i * s + j * f, where i,j are ints running over range `shape`
        mg = np.mgrid[0:shape[0]-1:1j*shape[0], 0:shape[1]-1:1j*shape[1]]
        xyz = np.outer(mg[0].flatten(), s) + np.outer(mg[1].flatten(), f)
        xyz += p # translate
        xyz = xyz.reshape( (shape[0], shape[1], 3) )

        return xyz
    
        
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
                raise TypeError('Explicit xyz pixels required, `xyz` must be np.ndarray')
            
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
                raise TypeError('Implicit xyz pixels required, `xyz` must be list')
            self._xyz_type = 'implicit'
            
            self._grid_list = xyz
            for g in self._grid_list:
                if (len(g) != 3) or (type(g) != tuple):
                    logger.critical("self._grid_list: %s" % str(self._grid_list))
                    raise ValueError('grid list not 3-tuple')
            
            self.num_pixels = int(np.sum([ b[1][0]*b[1][1]*b[1][2] for b in self._grid_list ]))
            self._pixels    = None
            
        else:
            raise TypeError("`xyz_type` must be one of {'explicit', 'implicit'}")
        
        
        # parse wavenumber
        if isinstance(k, Beam):
            self.k = k.wavenumber
            self.beam = k
        elif type(k) in [float, np.float64, np.float32]:
            self.k = k
            self.beam = None
        else:
            raise TypeError('`k` must be a float or odin.xray.Beam')
        
        self.path_length = path_length
        
        return
    
        
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
        
        
    @property
    def q_max(self):
        """
        Returns the maximum value of |q| the detector measures
        """
        if self.xyz_type == 'explicit':
            q_max = np.max(self.recpolar[:,0])
        elif self.xyz_type == 'implicit':
            q_max = 0.0
            for p in [self._grid_max(*t) for t in self._grid_list]:
                p[2] = self.path_length
                unit_p = p / self._norm(p)
                q_max_pt = self.k * (unit_p - np.array([0., 0., 1.]))
                qm = self._norm(q_max_pt)
                if qm > q_max: q_max = qm
        return q_max
        
        
    def _grid_max(self, basis, size, corner):
        """
        Find the (real-space) radial extremum of a grid. Returns: (x,y,z)
        """
        
        # find the coordinates of each corner
        # todo: add z
        bl = np.array(corner)
        br = bl + np.array([basis[0] * (size[0]-1), 0.0, 0.0])
        tl = bl + np.array([0.0, basis[1] * (size[1]-1), 0.0])
        tr = tl + np.array([basis[0] * (size[0]-1), 0.0, 0.0])
        
        # then the norm of each
        points = [bl, br, tl, tr]
        max_ind = np.argmax([ self._norm(p) for p in points ])
        r_max_pt = points[max_ind]
        
        return r_max_pt
        
        
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
    def generic(cls, spacing=1.00, lim=100.0, energy=10.0, 
                photons_scattered_per_shot=1e4, l=50.0, 
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
        
        beam = Beam(photons_scattered_per_shot, energy=energy)

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
            intensities = self._apply_polarization(intensities)
        
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
        mask[ np.abs(i - i.mean()) > cutoff ] = np.bool(False)
        return i, mask
        
        
    def polarization(self, polarization_factor, detector):
        """
        Remove the effects of beam polarization on the detected intensities.
        
        Parameters
        ----------
        polarization_factor : float
            
        """
        
        # TJL thinking : is passing a detector really the best way (most user 
        # friendly) to do this?
        
        self.qxyz = detector.reciprocal.copy()
        self.l = detector.path_length
        self._polarization_factor = polarization_factor
        self._methods_to_apply.append('polarization')
        
        return
        
        
    def _apply_polarization(self, i):
        """
        Apply the polarization filter to `i`
        """
        theta_x = np.arctan( self.qxyz[:,0] / self.l )
        theta_y = np.arctan( self.qxyz[:,1] / self.l )
        P = self._polarization_factor
        f_p = 0.5 * ( (1.0 + P) * np.power(np.cos(theta_x), 2) + \
                      (1.0 - P) * np.power(np.cos(theta_y), 2) )
        corrected_i = i.flatten() * f_p
        return corrected_i

        
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
        
    def __init__(self, intensities, detector, mask=None):
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
        mask : ndarray, np.bool
            An array the same size (and shape -- 1d) as `intensities` with a
            'np.True' in all indices that should be kept, and 'np.False'
            for all indices that should be masked.
        """
                
        self.intensities = intensities.flatten()
        self.detector    = detector
        
        if mask != None:
            if mask.shape != intensities.shape:
                raise ValueError('Mask must be same shape as intensities array!')
            self.mask = np.array(mask.flatten())
        else:
            self.mask = np.zeros(intensities.shape, dtype=np.bool)
            
        return
    
        
    @property
    def intensities_2d(self):
        return self.assemble_image()
    
        
    @staticmethod
    def num_phi_to_values(num_phi):
        """
        Converts `phi_spacing` to the explicit values, all in RADIANS.
        """
        phi_values = np.linspace(0.0, 2.0*np.pi, num=num_phi)
        return phi_values
    
        
    @staticmethod
    def num_phi_to_spacing(num_phi):
        return 2.0*np.pi / float(num_phi)
    
        
    def assemble_image(self, num_x=None, num_y=None):
        """
        Assembles the Shot object into a real-space image.

        Parameters
        ----------
        num_x,num_y : int
            The number of pixels in the x/y direction that will comprise the final
            grid.

        Returns
        -------
        grid_z : ndarray, float
            A 2d array representing the image one would see when viewing the
            shot in real space. E.g., to visualize w/matplotlib:

            >>> imshow(grid_z.T)
            >>> show()
            ...
        """

        if (num_x == None) or (num_y == None):
            # todo : better performance if needed (implicit detector)
            num_x = len(self.detector.xyz[:,0])
            num_y = len(self.detector.xyz[:,1])

        points = self.detector.xyz[:,:2] # ignore z-comp. of detector

        x = np.linspace(points[:,0].min(), points[:,0].max(), num_x)
        y = np.linspace(points[:,1].min(), points[:,1].max(), num_y)
        grid_x, grid_y = np.meshgrid(x,y)

        grid_z = interpolate.griddata(points, self.intensities, (grid_x,grid_y), 
                                      method='nearest', fill_value=0.0)

        return grid_z
    
        
    def _interpolate_to_polar(self, q_spacing=0.02, q_values=None, 
                              num_phi=1.0):
        """
        Interpolate our cartesian-based measurements into a polar coordiante 
        system.
        
        Parameters
        ----------
        q_spacing : float
            The q-vector spacing, in inverse angstroms.
            
        q_values : ndarray OR list OF floats
            If supplied, the interpolation will only be performed at these
            values of |q|, and the `q_spacing` parameter will be ignored.
            
        num_phi : int
            The number of equally-spaced points around the azimuth to
            interpolate (e.g. `num_phi`=360 means points at 1 deg spacing).
        
        Returns
        -------
        interpolated_intensities : ndarray, float
            The interpolated values.
            
        polar_mask : ndarray, bool
            A mask of ones and zeros. Ones are kept, zeros masked.
        """
        
        # compute q_values if need be
        if q_values != None:
            q_values = np.array(q_values)
        else:
            q_min     = q_spacing
            q_max     = self.detector.q_max
            q_values  = np.arange(q_min, q_max, q_spacing)
        
        # check to see what method we want to use to interpolate. Here,
        # `unstructured` is more general, but slower; implicit/structured assume
        # the detectors are grids, and are therefore faster but specific
        
        if self.detector.xyz_type == 'explicit':
            polar_intensities, polar_mask = self._unstructured_interpolation(q_values, num_phi)
        elif self.detector.xyz_type == 'implicit':
            polar_intensities, polar_mask = self._implicit_interpolation(q_values, num_phi)
        else:
            raise RuntimeError('Invalid detector passed to Shot(), must be of '
                               'xyz_type {explicit, implicit}')
                               
        # finally, be sure we include any pixels explicitly masked in cart.
        # space
        polar_mask = self._apply_mask_to_polar_grid(polar_mask)
        
        return polar_intensities, polar_mask
    
        
    def polar_grid(self, q_values, num_phi):
        """
        Return the pixels that comprise the polar grid in (q, phi) space.
        """
        
        phi_values = self.num_phi_to_values(num_phi)
        num_q = len(q_values)

        polar_grid = np.zeros((num_q * num_phi, 2))
        polar_grid[:,0] = np.tile(q_values, num_phi)
        polar_grid[:,1] = np.repeat(phi_values, num_q)

        return polar_grid
    

    def polar_grid_as_cart(self, q_values, num_phi):
        """
        Returns the pixels that comprise the polar grid in q-cartesian space,
        (q_x, q_y)
        """
        
        phi_values = self.num_phi_to_values(num_phi)
        num_q = len(q_values)
        
        pg_real = np.zeros((num_q * num_phi, 2))
        pg_real[:,0] = np.tile(q_values, num_phi) * np.cos( np.repeat(phi_values, num_q) )
        pg_real[:,1] = np.tile(q_values, num_phi) * np.sin( np.repeat(phi_values, num_q) )

        return pg_real
    

    def polar_grid_as_real_cart(self, q_values, num_phi):
        """
        Returns the pixels that comprise the polar grid in real cartesian space,
        (x, y)
        """
        
        phi_values = self.num_phi_to_values(num_phi)
        num_q = len(q_values)
        
        k  = self.detector.k
        l  = self.detector.path_length
        
        # the real-space radius of each pixel, but "untiled" to save some mem.
        h = l * np.tan( 2.0 * np.arcsin( (q_values+1.0e-300) / (2.0*k) ) )
        
        phis = np.repeat(phi_values, num_q)
        
        pg = np.zeros((num_q * num_phi, 2))
        pg[:,0] = np.tile(h, num_phi) * np.cos( phis )
        pg[:,1] = np.tile(h, num_phi) * np.sin( phis )
                        
        return pg
        
        
    @staticmethod
    def _overlap(xy1, xy2):
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
        
        
    @staticmethod
    def _overlap_implicit(xy, grid):
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
       
        p_inds  = np.where(( xy[:,0] > min_x ) * ( xy[:,0] < max_x ) *\
                           ( xy[:,1] > min_y ) * ( xy[:,1] < max_y ))[0]
        
        return p_inds
        
        
    def _implicit_interpolation(self, q_values, num_phi):
        """
        Interpolate onto a polar grid from an `implicit` detector geometry.
        
        This detector geometry is specified by the x and y pixel spacing,
        the number of pixels in the x/y direction, and the top-left corner
        position.
        
        NOTE: The interpolation is performed in cartesian real (xyz) space.
        """
        
        # initialize output space for the polar data and mask
        num_polar_points  = len(q_values) * num_phi
        polar_intensities = np.zeros(num_polar_points)
        polar_mask        = np.zeros(num_polar_points, dtype=np.bool)
                
        # loop over all the arrays that comprise the detector and use
        # grid interpolation on each
        
        int_start = 0 # start of intensity array correpsonding to `grid`
        int_end   = 0 # end of intensity array correpsonding to `grid`
        
        # convert the polar to cartesian coords for comparison to detector
        pgr = self.polar_grid_as_real_cart(q_values, num_phi)
        
        # iterate over each grid area
        for k,grid in enumerate(self.detector._grid_list):
            
            basis, size, corner = grid
            n_int = int(size[0] * size[1])
            int_end += n_int
        
            # sanity checks
            if size[2] != 1:
                logger.debug('Warning: Z dimension not flat')
        
            if (basis[1] == 0.0) or (basis[0] == 0.0):
                raise RuntimeError('Implicit detector is flat in either x or y!'
                                   ' Cannot be interpolated.')
            
            # find the indices of the polar grid pixels that are inside the
            # boundaries of the current detector array we're interpolating over
            p_inds = self._overlap_implicit(pgr, grid)
            
            if len(p_inds) == 0:
                logger.warning('Detector array (%d/%d) had no pixels inside the \
                interpolation area!' % (k+1, len(self.detector.grid_list)) )
                continue
                        
            # interpolate onto the polar grid & update the inverse mask
            # perform the interpolation. 
            interp = Bcinterp(self.intensities[int_start:int_end], 
                              basis[0], basis[1], size[0], size[1], 
                              corner[0], corner[1])
            
            polar_intensities[p_inds] = interp.evaluate(pgr[p_inds,0], 
                                                        pgr[p_inds,1])
            polar_mask[p_inds] = np.bool(True)
            
            # increment index for self.intensities -- the real/measured intsn.
            int_start += n_int
        
        return polar_intensities, polar_mask
        
        
    def _unstructured_interpolation(self, q_values, num_phi):
        """
        Perform an interpolation to polar coordinates assuming that the detector
        pixels do not form a rectangular grid.
        
        NOTE: The interpolation is performed in polar momentum (q) space.
        """
        
        # initialize output space for the polar data and mask
        num_polar_points  = len(q_values) * num_phi
        polar_intensities = np.zeros(num_polar_points)
        polar_mask        = np.zeros(num_polar_points, dtype=np.bool)
        
        # interpolate onto polar grid : delauny triangulation + linear
        polar = self.detector.polar
        xy = np.zeros(( polar.shape[0], 2 ))
        xy[:,0] = self.detector.k * np.sqrt( 2.0 - 2.0 * np.cos(polar[:,1]) ) # |q|
        xy[:,1] = polar[:,2] # phi
        
        # because we're using a "square" interplation method, wrap around one
        # set of polar coordinates to capture the periodic nature of polar coords
        add = np.where( xy[:,1] == xy[:,1].min() )[0]
        xy_add = xy[add]
        xy_add[:,1] += 2.0 * np.pi
        aug_xy  = np.concatenate(( xy, xy_add ))
        aug_int = np.concatenate(( self.intensities, self.intensities[add] ))
        
        # do the interpolation
        z_interp = interpolate.griddata( aug_xy, aug_int, self.polar_grid(q_values, num_phi),
                                         method='linear', fill_value=np.nan )

        self.polar_intensities = z_interp.flatten()

        # mask missing pixels (outside convex hull)
        nan_ind = np.where( np.isnan(z_interp) )[0]
        polar_intensities[nan_ind] = 0.0
        not_nan_ind = np.delete(np.arange(z_interp.shape[0]), nan_ind)
        
        polar_mask[not_nan_ind] = np.bool(True)
        
        return polar_intensities, polar_mask
                
        
    def _apply_mask_to_polar_grid(self, polar_mask):
        """
        Sets the polar mask to mask the pixels indicated by the mask argument.
        
        NOTE : For all ODIN masks, one/True is "good", zero/False is "masked"
        """
        
        # if we have no work to do, let's escape!
        if np.sum(self.mask) == np.product(self.mask.shape):
            return
        
        # perform a fixed-radius nearest-neighbor search, masking all the pixels
        # on the polar grid that are within one-half pixel spacing from a masked
        # pixel on the cartesian grid
        
        xyz = self.detector.reciprocal
        pgc = self.polar_grid_as_cart(q_values, num_phi)
        
        # the max distance, r^2 - a factor of 10 helps get the spacing right
        r2 = np.sum( np.power( xyz[0] - xyz[1], 2 ) ) * 10.0
        masked_cart_points = xyz[self.mask,:2]
        
        # todo : currently slow, can be done faster? Could Weave + OMP
                
        # loop over masked cartesian points and mask polar pixels that are close
        for mp in masked_cart_points:
            d2 = np.sum( np.power( mp - pgc[:,:2], 2 ), axis=1 )
            polar_mask[ d2 >= r2 ] = np.bool(False)
        
        return polar_mask
    

    def intensity_profile(self):
        """
        Averages over the azimuth phi to obtain an intensity profile.
        
        Returns
        -------
        intensity_profile : ndarray, float
            An n x 2 array, where the first dimension is the magnitude |q| and
            the second is the average intensity at that point < I(|q|) >_phi.
        """
         
        # todo : re-write
        raise NotImplementedError()

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
        
        
    @classmethod
    def simulate(cls, traj, num_molecules, detector, traj_weights=None, 
                 finite_photon=False, force_no_gpu=False, device_id=0):
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
            
        finite_photon : bool
            Use finite photon statistics in the simulation.
    
        force_no_gpu : bool
            Run the (slow) CPU version of this function.
 
        device_id : int
            The index of the GPU to run on.
    
        Returns
        -------
        shot : odin.xray.Shot
            A shot instance, containing the simulated shot.
        """
        I = scatter.simulate_shot(traj, num_molecules, detector, 
                                  traj_weights=traj_weights,
                                  finite_photon=finite_photon, 
                                  force_no_gpu=force_no_gpu,
                                  device_id=device_id)
        shot = Shot(I, detector)
        return shot
    
    # So as to not duplicate code/logic, I am going to employ a wrapped version
    # of the Shotset save/load/to_rings methods here...
    
    def to_rings(self, q_values=None, num_phi=360):
        """
        Convert the shot to an xray.Rings object, for computing correlation
        functions and other properties in polar space.
        
        This automatically interpolates the dataset onto a polar grid and then
        converts those polar values into a class that facilitates computation
        in that space. See odin.xray.Rings for more info.
        
        Parameters
        ----------
        q_values : ndarray/list, float
            The values of |q| to extract rings at (in Ang^{-1}). If `None`, then
            will simply fill up a range of values at |q|=0.02 spacing. Recommend
            you choose something here!
            
        num_phi : int
            The number of equally spaced points around the azimuth to 
            interpolate onto (e.g. `num_phi`=360 means 1 deg spacing).
        """
        ss = Shotset([self])
        return ss.to_rings(q_values, num_phi)
    
        
    def save(self, filename):
        """
        Writes the current Shot data to disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file to save.
        """
        ss = Shotset([self])
        ss.save(filename)
        return


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


class Shotset(object):
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
            raise TypeError('Cannot add types: %s and Shotset' % type(other))
        self.shots.extend(other.shots)
        assert len(self.shots) > 0
        return Shotset( self.shots )
        
        
    def _check_qvectors_same(self, epsilon=1e-6):
        """
        For a lot of the below to work, we need to make sure that all of the 
        q-phi grids in the shots are the same. If some are different, here
        we recalculate them.
        """

        # todo
        return
        
        #q_phis = self.shots[0].polar_grid
        #for shot in self.shots:
        #    diff = np.sum(np.abs(shot.polar_grid - q_phis))
        #    if diff > epsilon:
        #        raise ValueError('Detector values in Shotset not consistent '
        #                          ' across set. Homogenize interpolated polar'
        #                          ' grid using Shot.interpolate_to_polar()')
    
        
    def intensity_profile(self):
        """
        Averages over the azimuth phi to obtain an intensity profile.

        Returns
        -------
        intensity_profile : ndarray, float
            An n x 2 array, where the first dimension is the magnitude |q| and
            the second is the average intensity at that point < I(|q|) >_phi.
        """
        
        # todo : rewrite
        raise NotImplementedError()
        
        return intensity_profile
    
        
    def to_rings(self, q_values=None, num_phi=360):
        """
        Convert the shot to an xray.Rings object, for computing correlation
        functions and other properties in polar space.

        This automatically interpolates the dataset onto a polar grid and then
        converts those polar values into a class that facilitates computation
        in that space. See odin.xray.Rings for more info.

        Parameters
        ----------
        q_values : ndarray/list, float
            The values of |q| to extract rings at (in Ang^{-1}). If `None`, then
            will simply fill up a range of values at |q|=0.02 spacing. Recommend
            you choose something here!

        num_phi : int
            The number of equally spaced points around the azimuth to 
            interpolate onto (e.g. `num_phi`=360 means 1 deg spacing).
        """
        
        logger.info('Converting %d shots to polar space (Rings)' % self.num_shots)
        
        num_q = len(q_values)
        
        # initialize arrays
        polar_intensities = np.zeros((self.num_shots, num_q, num_phi))
        polar_mask = np.zeros((self.num_shots, num_q, num_phi), dtype=np.bool)
        
        # loop over all shots and interpolate them to polar
        for i in range(self.num_shots):
            
            logger.debug('Shot: %d/%d' % (i, self.num_shots))
            
            # below if q_values == None, will interpolate at spacing |q| = 0.02
            pi, pm = self.shots[i]._interpolate_to_polar(q_values=q_values, 
                                                         num_phi=num_phi)
            polar_intensities[i,:,:] = pi.reshape(num_q, num_phi)
            polar_mask[i,:,:]        = pm.reshape(num_q, num_phi)
        
        r = Rings(q_values, polar_intensities, beam.k, polar_mask)
        
        return r
    
                    
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
            shotdata[('shot%d_mask' % i)] = self.shots[i].mask

        io.saveh(filename, 
                 num_shots = np.array([self.num_shots]),
                 detector  = self.shots[0].detector._to_serial(),
                 **shotdata)

        logger.info('Wrote %s to disk.' % filename)
        
        return
    

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

            num_shots = int(hdf['num_shots'])
            
            # figure out which shots to load
            if to_load == None:
                to_load = range(num_shots)
            else:
                try:
                    to_load = np.array(to_load)
                except:
                    raise TypeError('`to_load` must be a ndarry/list of ints')
        
            list_of_shots = []
            for i in to_load:
                i = int(i)
                intensities = hdf[('shot%d' % i)]
                d = Detector._from_serial(hdf['detector'])
                
                # some older files may not have any mask, so be gentle
                if ('shot%d_mask' % i) in hdf.keys():
                    mask = hdf[('shot%d_mask' % i)]
                else:
                    mask = None
                
                s = Shot(intensities, d, mask=mask)
                list_of_shots.append(s)
            
        elif filename.endswith('.cxi'):
            raise NotImplementedError() # todo
            
        else:
            raise ValueError('Must load a shotset file [.shot, .cxi]')

        hdf.close()

        return cls(list_of_shots)
        
        
class Rings(object):
    """
    Class to keep track of intensity data in a polar space.
    """
    
    def __init__(self, q_values, polar_intensities, k, polar_mask=None):
        """
        Interpolate our cartesian-based measurements into a polar coordiante 
        system.
        
        Parameters
        ----------
        q_values : ndarray OR list OF floats
            The values of |q| in `polar_intensities`, in inverse Angstroms.
        
        polar_intensities : ndarray, float
            Intensities in polar space. Should be shape:
            
                N x len(`q_values`) x num_phi
            
            with N the number of shots (any value) and `num_phi` the number of
            points (equally spaced) around the azimuth.
            
        k : float
            The wavenumber of the energy used to acquire the data.
            
        polar_mask : ndarray, bool
            A mask of ones and zeros. Ones are kept, zeros masked. Should be the
            same shape as `polar_intensities`. Can also be `None`, meaning no
            masked pixels
        """
        
        if not polar_intensities.shape[1] == len(q_values):
            raise ValueError('`polar_intensities` must have same len as '
                             '`q_values` in its second dimension.')
        
        if polar_mask == None:
            self.polar_mask = None
        elif type(polar_mask) == np.ndarray:
            if not polar_intensities.shape == polar_mask.shape:
                raise ValueError('`polar_mask` must have same shape as '
                                 '`polar_intensities')
            self.polar_mask = None
        else:
            raise TypeError('`polar_mask` must be np.ndarray or None')
            
        self._q_values = q_values
        self.polar_intensities = polar_intensities
        self.k = k
        
        return
    
        
    @property
    def num_shots(self):
        return self.polar_intensities.shape[0]
    
        
    @property
    def phi_values(self):
        return np.linspace(0.0, 2.0*np.pi, num=self.num_phi)
        
        
    @property
    def q_values(self):
        return self._q_values
    
        
    @property
    def num_phi(self):
        return self.polar_intensities.shape[2]
    

    @property
    def num_q(self):
        return len(self._q_values)
    

    @property
    def num_datapoints(self):
        return self.num_phi * self.num_q
    
    
    def q_index(self, q):
        """
        Convert value of |q| (in inverse Angstroms) into the index used to
        slice `polar_intensities`.
        
        Parameters
        ----------
        q : float
            The value of |q| in inv Angstroms
            
        Returns
        -------
        q_ind : int
            The index to slice `polar_intensities` at to get |q|.
        """
        q_ind = int( np.where( self.q_values == q )[0] )
        return q_ind
    
        
    def intensity_profile(self):
        """
        Averages over the azimuth phi to obtain an intensity profile.

        Returns
        -------
        intensity_profile : ndarray, float
            An n x 2 array, where the first dimension is the magnitude |q| and
            the second is the average intensity at that point < I(|q|) >_phi.
        """
        
        intensity_profile = zeros( (len(ip), 2), dtype=np.float )
        intensity_profile[:,0] = self._q_values.copy()
        
        # average over shots, phi
        intensity_profile[:,1] = np.mean( np.mean(self.polar_intensities * \
                             self.polar_mask.asdtype(np.float), axis=2), axis=0)
        
        return intensity_profile
    
        
    def correlate(self, q1, q2, delta_ind):
        """
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi.
    
        Parameters
        ----------
        q1 : float
            The magnitude of the first position to correlate.
        q2 : float
            The magnitude of the second position to correlate.
        delta_ind : int
            The index of the angle between the first and second q-vectors to 
            calculate the correlation for.
        
        Returns
        -------
        correlation : float
            The correlation between q1/q2 at angle delta. Specifically, this
            is the correlation function with the mean subtracted <x*y> - <x><y>
        """
        q_ind1 = self.q_index(q1)
        q_ind2 = self.q_index(q2)
        return self._correlate_by_index(q_ind1, q_ind2, delta_ind)
    

    def _correlate_by_index(self, q_ind1, q_ind2, delta_ind):
        """
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi. 
        
        Here the arguments are the raw indices to slice self.polar_intensities
        
        Parameters
        ----------
        q_ind1 : int
            The index of the magnitude of the first position to correlate.
        q_ind2 : int
            The index of the magnitude of the second position to correlate.
        delta_ind : int
            The index of the angle between the first and second q-vectors to 
            calculate the correlation for.
        
        Returns
        -------
        correlation : float
            The correlation between q1/q2 at angle delta. Specifically, this
            is the correlation function with the mean subtracted <x*y> - <x><y>
        """
        
        q_ind1 = int(q_ind1)
        q_ind2 = int(q_ind2)
        
        x = self.polar_intensities[:,q_ind1,:]
        y = self.polar_intensities[:,q_ind2,:]
        y = np.roll(y, delta_ind, axis=1)

        # going to use np.ma below
        if self.polar_mask:
            x = np.ma.masked_array( x, mask=(np.bool(True) - self.polar_mask[:,q_ind1,:]) )
            y = np.ma.masked_array( y, mask=(np.bool(True) - self.polar_mask[:,q_ind2,:]) )

        # this should work w/masking
        x -= x.mean()
        y -= y.mean()

        corr = float( np.mean(x * y) / (x.std() * y.std()) )

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
        """
        
        logger.debug("Correlating rings at %f / %f" % (q1, q2))
        
        q_ind1 = self.q_index(q1)
        q_ind2 = self.q_index(q2)
        
        return self._correlate_ring_by_index(q_ind1, q_ind2)
    
        
    def _correlate_ring_by_index(self, q_ind1, q_ind2):
        """
        Compute the correlation function C(q1, q2, delta) for the shot, averaged
        for each measured value of the azimuthal angle phi, for many values
        of delta.
        
        Parameters
        ----------
        q_ind1 : int
            The index of the magnitude of the first position to correlate.
        q_ind2 : int
            The index of the magnitude of the second position to correlate.
            
        Returns
        -------
        correlation_ring : ndarray, float
            A 2d array, where the first dimension is the value of the angle
            delta employed, and the second is the correlation at that point.
        """
        
        q_ind1 = int(q_ind1)
        q_ind2 = int(q_ind2)

        x = self.polar_intensities[:,q_ind1,:]
        y = self.polar_intensities[:,q_ind2,:]

        # As DTM has pointed out, here we need to be a little careful with
        # how we deal with masked pixels. If we subtract the means from the
        # pixels first, then perform the correlation, we can safely ignore
        # masked pixels so long as we make sure their value is zero
                
        if self.polar_mask:
            x = np.ma.masked_array( x, mask=(np.bool(True) - self.polar_mask[:,q_ind1,:]) )
            y = np.ma.masked_array( y, mask=(np.bool(True) - self.polar_mask[:,q_ind2,:]) )

        x -= x.mean()
        y -= y.mean()

        # plug in a zero at all mask positions
        if np.ma.isMaskedArray(x):
            x[x.mask] = 0.0
        if np.ma.isMaskedArray(y):
            y[y.mask] = 0.0

        # use d-FFT + convolution thm
        ffx = fftpack.fft(x, axis=1)
        ffy = fftpack.fft(y, axis=1)
        iff = np.real(fftpack.ifft( ffx * np.conjugate(ffy), axis=1 ))

        # gotta loop over the shots here and average them (norm is STD, faster)
        # correlation_ring = np.zeros(self.num_phi)
        # for i in range(self.num_shots):
        #     correlation_ring[:] += iff[i,:] / (np.linalg.norm(x[i,:]) * \
        #                                        np.linalg.norm(y[i,:]))
        # correlation_ring[:] /= float(self.num_shots)
        
        correlation_ring = iff.sum(axis=0)
        correlation_ring /= correlation_ring[0]

        return correlation_ring
    
        
    def legendre(self, order, report_error=False):
        """
        Project the correlation functions onto a set of legendre polynomials,
        and return the coefficients of that projection.

        Parameters
        ----------
        order : int
            The order at which to truncate the polynomial expansion. Note that
            this function projects only onto even numbered Legendre polynomials.
            
        Returns
        -------
        Cl: np.ndarray, float
            An array of the legendre coefficients. Contains all coefficients
            (both even and odd) up to order `order`. The returned object is a
            3-D array indexed by
            
                (order, q_ind1, q_ind2)
                
            where the q_ind values are the indices that map onto self.q_values.
        """
        
        # todo : use Friedel pairs to compute entire Kam correlator (?)

        # initialize space for coefficients        
        Cl = np.zeros((order, self.num_q, self.num_q))

        # iterate over each pair of rings in the collection and project the
        # correlation between those two rings into the Legendre basis

        for i in range(self.num_q):
            for j in range(i,self.num_q):

                # compute psi, the angle between the scattering vectors
                t1 = np.arctan( self.q_values[i] / (2.0*self.k) ) # theta1
                t2 = np.arctan( self.q_values[j] / (2.0*self.k) ) # theta2

                # compute the Kam scattering angle psi
                psi = np.arccos( np.cos(t1)*np.cos(t2) + np.sin(t1)*np.sin(t2)*\
                                 np.cos( self.phi_values * 2. * \
                                 np.pi/float(self.num_phi) ) )

            	# tests indicate this method is a good way of doing the 
            	# numerical projection
                c, fit_data = np.polynomial.legendre.legfit(np.cos(psi), 
                                        self._correlate_ring_by_index(i,j), 
                                        order-1, full=True)

                Cl[:,i,j] = c
                Cl[:,j,i] = c  # copy it to the lower triangle too   

        return Cl


    @classmethod
    def simulate(cls, traj, num_molecules, q_values, num_phi, num_shots,
                 energy=10, traj_weights=None, finite_photon=False, 
                 force_no_gpu=False, photons_scattered_per_shot=1e4, 
                 device_id=0):
        """
        Simulate many scattering 'shot's, i.e. one exposure of x-rays to a 
        sample, but onto a polar detector. Return that as a Rings object 
        (factory function).

        Assumes we have a Boltzmann distribution of `num_molecules` identical  
        molecules (`trajectory`), exposed to a beam defined by `beam` and 
        projected onto `detector`.

        Each conformation is randomly rotated before the scattering simulation is
        performed. Atomic form factors from X, finite photon statistics, and the 
        dilute-sample (no scattering interference from adjacent molecules) 
        approximation are employed.

        Parameters
        ----------
        traj : mdtraj.trajectory
            A trajectory object that contains a set of structures, representing
            the Boltzmann ensemble of the sample. If len(traj) == 1, then we 
            assume the sample consists of a single homogenous structure,
            replecated `num_molecules` times.

        q_values : ndarray/list, float
            The values of |q| to extract rings at (in Ang^{-1}).

        num_phi : int
            The number of equally spaced points around the azimuth to 
            interpolate onto (e.g. `num_phi`=360 means 1 deg spacing).
                
        num_molecules : int
            The number of molecules estimated to be in the `beam`'s focus.

        num_shots : int
            The number of shots to perform and include in the Shotset.

        Optional Parameters
        -------------------
        energy : float
            The energy, in keV
        
        traj_weights : ndarray, float
            If `traj` contains many structures, an array that provides the 
            Boltzmann weight of each structure. Default: if traj_weights == None
            weights each structure equally.

        finite_photon : bool
            Whether or not to employ finite photon statistics in the simulation

        force_no_gpu : bool
            Run the (slow) CPU version of this function.
            
        photons_scattered_per_shot : int
            The number of photons scattered to the detector per shot. For use
            with `finite_photon`.

        Returns
        -------
        rings : odin.xray.Rings
            A Rings instance, containing the simulated shots.
        """

        device_id = int(device_id)
        beam = Beam(photons_scattered_per_shot, energy=energy)
        k = beam.k
            
        # --- generate the polar grid ---        
        
        phi_values = np.linspace(0.0, 2.0*np.pi, num=num_phi )
        num_q = len(q_values)
        
        # q_h is the magnitude projection of the scattering vector on (x,y)
        q_h = np.power(q_values, 2) / (2.0 * k)
        q_z = q_values * np.sqrt( 1.0 - np.power( q_values / (2.0 * k), 2 ) )
            
        # construct the polar grid:
        #   gotta tile q / repeat phi to make the reshape below work right
        #   this is weird but consistent with other parts of the code in Shot
        qxyz = np.zeros(( num_q * num_phi, 3 ))
        qxyz[:,0] = np.tile(q_h, num_phi) * np.cos(np.repeat(phi_values, num_q)) # q_x
        qxyz[:,1] = np.tile(q_h, num_phi) * np.sin(np.repeat(phi_values, num_q)) # q_y
        qxyz[:,2] = np.tile( q_z, num_phi )                                      # q_z
        
        # assert that the above vectors are the correct length
        assert np.all(np.sqrt( np.sum( np.power(qxyz,2), axis=1 ) ) == np.tile(q_values, num_phi))
        
        # --- simulate the intensities ---
        
        polar_intensities = np.zeros((num_shots, num_q, num_phi))
        
        for i in range(num_shots):
            I = scatter.simulate_shot(traj, num_molecules, qxyz, 
                                      traj_weights=traj_weights,
                                      finite_photon=finite_photon,
                                      force_no_gpu=force_no_gpu,
                                      device_id=device_id)
            polar_intensities[i,:,:] = I.reshape(num_phi, len(q_values)).T

            logger.info('Finished polar shot %d/%d on device %d' % (i+1, num_shots, device_id) )

        return cls(q_values, polar_intensities, k, polar_mask=None)
        
        
    def save(self, filename):
        """
        Saves the Rings object to disk.
        
        Parameters
        ----------
        filename : str
            The name of the file to write to disk. Must end in '.ring' -- if you
            don't put this, it will be automatically added.
        """
        
        if not filename.endswith('.ring'):
            filename += '.ring'

        # if self.polar_mask == None, then save a single 0
        if self.polar_mask == None:
            pm = np.array([0])
        else:
            pm = self.polar_mask

        io.saveh( filename,
                  q_values = self._q_values,
                  polar_intensities = self.polar_intensities,
                  k = np.array([self.k]),
                  polar_mask = pm )

        logger.info('Wrote %s to disk.' % filename)
        
        return
    
        
    @classmethod
    def load(cls, filename):
        """
        Load a Rings object from disk.
        
        Parameters
        ----------
        filename : str
            The name of the file to write to disk. Must end in '.ring'.
        """
        
        if filename.endswith('.ring'):
            hdf = io.loadh(filename)
        else:
            raise ValueError('Must load a rings file (.ring)')

        # deal with our codified polar mask
        if np.all(hdf['polar_mask'] == np.array([0])):
            pm = None
        else:
            pm = hdf['polar_mask']
            
        rings_obj = cls(hdf['q_values'], hdf['polar_intensities'], 
                        float(hdf['k'][0]), polar_mask=pm)
        hdf.close()

        return rings_obj
    
    
        
