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
from matplotlib import nxutils
from scipy import interpolate, fftpack
from scipy.ndimage import filters
from scipy.special import legendre

from odin.math2 import arctan3, smooth
from odin import scatter
from odin.interp import Bcinterp
from odin.utils import unique_rows, maxima, random_pairs
from odin.corr import correlate as gap_correlate

from mdtraj import trajectory, io

from odin import corr

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
    a single rectangular pixel grid lives in. They may also be called the y and
    x dimensions without any loss of generality.

    The convention here -- and in all of ODIN -- is one of Row-Major ordering,
    which is consistent with C/python. This means that y is the slow dim, x is
    the fast dim, and when ordering these two they will appear as (slow, fast).

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

        self._num_grids = 0
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


    @property
    def num_pixels(self):
        """
        Return the total number of pixels in the BasisGrid.
        """
        n = np.sum([np.product(self._shapes[i]) for i in range(self.num_grids)])
        return int(n)


    @property
    def num_grids(self):
        return self._num_grids


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
        self._num_grids += 1
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


    def get_grid_corners(self, grid_number):
        """
        Return the positions of the four corners of a grid.

        Parameters
        ----------
        grid_number : int
            The index of the grid to get the corners of.

        Returns
        -------
        corners : np.ndarray, float
            A 4 x 3 array, where the first dim represents the four corners, and
            the second is x/y/z. Note one corner is always just the `p` vector.
        """

        if grid_number >= self.num_grids:
            raise ValueError('Only %d grids in object, you asked for the %d-th'
                             ' (zero indexed)' % (self.num_grids, grid_number))

        # compute the lengths of the parallelogram sides
        s_side = self._fs[grid_number] * float(self._shapes[grid_number][0])
        f_side = self._ss[grid_number] * float(self._shapes[grid_number][1])
        pc = self._ps[grid_number].copy()

        corners = np.zeros((4,3))

        corners[0,:] = pc
        corners[1,:] = pc + s_side
        corners[2,:] = pc + f_side
        corners[3,:] = pc + s_side + f_side

        return corners


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

    def __init__(self, xyz, k, beam_vector=None):
        """
        Instantiate a Detector object.

        Detector objects provide a handle for the many representations of
        detector geometry in scattering experiments, namely:

        -- real space
        -- real space in polar coordinates
        -- reciprocal space (q-space)
        -- reciprocal space in polar coordinates (q, theta, phi)

        Note the the origin is assumed to be the interaction site.

        Parameters
        ----------
        xyz : ndarray OR xray.BasisGrid
            An a specification the (x,y,z) positions of each pixel. This can
            either be n x 3 array with the explicit positions of each pixel,
            or a BasisGrid object with a vectorized representation of the
            pixels. The latter yeilds higher performance, and is recommended.

        k : float or odin.xray.Beam
            The wavenumber of the incident beam to use. Optionally a Beam
            object, defining the beam energy.

        Optional Parameters
        -------------------
        beam_vector : float
            The 3-vector describing the beam direction. If `None`, then the
            beam is assumed to be purely in the z-direction.
        """

        if type(xyz) == np.ndarray:
            logger.debug('xyz type: np.ndarray, initializing an explicit detector')

            self._pixels = xyz
            self._basis_grid = None
            self.num_pixels = xyz.shape[0]
            self._xyz_type = 'explicit'

        elif type(xyz) == BasisGrid:
            logger.debug('xyz type: BasisGrid, initializing an implicit detector')

            self._pixels = None
            self._basis_grid = xyz
            self.num_pixels = self._basis_grid.num_pixels
            self._xyz_type = 'implicit'

        else:
            raise TypeError("`xyz` type must be one of {'np.ndarray', "
                            "'odin.xray.BasisGrid'}")


        # parse wavenumber
        if isinstance(k, Beam):
            self.k = k.wavenumber
            self.beam = k
        elif type(k) in [float, np.float64, np.float32]:
            self.k = k
            self.beam = None
        else:
            raise TypeError('`k` must be a float or odin.xray.Beam')

        # parse beam_vector -- is guarenteed to be a unit vector
        if beam_vector != None:
            if beam_vector.shape == (3,):
                self.beam_vector = self._unit_vector(beam_vector)
            else:
                raise ValueError('`beam_vector` must be a 3-vector')
        else:
            self.beam_vector = np.array([0.0, 0.0, 1.0])

        return


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
            return self._basis_grid.to_explicit()


    @property
    def real(self):
        return self.xyz.copy()


    @property
    def polar(self):
        return self._real_to_polar(self.real)


    @property
    def reciprocal(self):
        return self._real_to_reciprocal(self.real)


    @property
    def recpolar(self):
        a = self._real_to_recpolar(self.real)
        # convention: theta is angle of q-vec with plane normal to beam
        a[:,1] = self.polar[:,1] / 2.0
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
            for i in range(self._basis_grid.num_grids):
                c  = self._basis_grid.get_grid_corners(i)
                qc = self._real_to_recpolar(c)
                q_max = max([q_max, float(np.max(qc[:,0]))])

        return q_max


    def evaluate_qmag(self, xyz):
        """
        Given the positions of pixels `xyz`, compute the corresponding |q|
        value for each.

        Parameters
        ----------
        qxyz : ndarray, float
            The array of pixels (shape : N x 3)

        Returns
        -------
        qmag : ndarray, float
            The array of q-magnitudes, len N.
        """
        thetas = self._evaluate_theta(xyz)
        qmag = 2.0 * self.k * np.sin(thetas/2.0)
        return qmag


    def _evaluate_theta(self, xyz):
        """
        Given the positions of pixels `xyz`, compute the corresponding
        scattering angle theta for each.

        Parameters
        ----------
        xyz : ndarray, float
            The array of pixels (shape : N x 3)

        Returns
        -------
        thetas : ndarray, float
            The scattering angles for each pixel
        """
        u_xyz  = self._unit_vector(xyz)
        thetas = np.arccos(np.dot( u_xyz, self.beam_vector ))
        return thetas


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
        S = self._unit_vector(xyz)
        q = self.k * (S - self.beam_vector)

        return q


    def _real_to_recpolar(self, xyz):
        """
        Convert the real-space to reciprocal-space in polar form, that is
        (|q|, theta , phi).
        """
        reciprocal_polar = self._to_polar( self._real_to_reciprocal(xyz) )
        return reciprocal_polar


    @staticmethod
    def _norm(vector):
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

        if len(vector.shape) == 1:
            unit_vectors = vector / norm

        elif len(vector.shape) == 2:
            unit_vectors = np.zeros( vector.shape )
            for i in range(vector.shape[0]):
                unit_vectors[i,:] = vector[i,:] / norm[i]

        else:
            raise ValueError('invalid shape for `vector`: %s' % str(vector.shape))

        return unit_vectors


    def _to_polar(self, vector):
        """
        Converts n m-dimensional `vector`s to polar coordinates. By polar
        coordinates, I mean the cannonical physicist's (r, theta, phi), no
        2-theta business.

        We take, as convention, the 'z' direction to be along self.beam_vector
        """

        polar = np.zeros( vector.shape )

        # note the below is a little modified from the standard, to take into
        # account the fact that the beam may not be only in the z direction

        polar[:,0] = self._norm(vector)
        polar[:,1] = np.arccos( np.dot(vector, self.beam_vector) / \
                                (polar[:,0]+1e-16) )           # cos^{-1}(z.x/r)
        polar[:,2] = arctan3(vector[:,1] - self.beam_vector[1],
                             vector[:,0] - self.beam_vector[0])   # y first!

        return polar


    def _compute_intersections(self, q_vectors, grid_index, run_checks=True):
        """
        Compute the points i=(x,y,z) where the scattering vectors described by
        `q_vectors` intersect the detector.

        Parameters
        ----------
        q_vectors : np.ndarray
            An N x 3 array representing q-vectors in cartesian q-space.

        grid_index : int
            The index of the grid array to intersect

        Optional Parameters
        -------------------
        run_checks: bool
            Whether to run some good sanity checks, at small computational cost.

        Returns
        -------
        pix_n : ndarray, float
            The coefficients of the position of each intersection in terms of
            the basis grids s/f vectors.

        intersect: ndarray, bool
            A boolean array of which of `q_vectors` intersect with the grid
            plane. First column is slow scan vector (s),  second is fast (f).

        References
        ----------
        .[1] http://en.wikipedia.org/wiki/Line-plane_intersection
        """

        if not self.xyz_type == 'implicit':
            raise RuntimeError('intersections can only be computed for implicit'
                               ' detectors')

        # compute the scattering vectors correspoding to q_vectors
        S = (q_vectors / self.k) + self.beam_vector

        # compute intersections
        p, s, f, shape = self._basis_grid.get_grid(grid_index)
        n = self._unit_vector( np.cross(s, f) )
        i = (np.dot(p, n) / np.dot(S, n))[:,None] * S

        # convert to pixel units by solving for the coefficients of proj
        A = np.array([s,f]).T
        pix_n, resid, rank, sigma = np.linalg.lstsq( A, (i-p).T )

        if run_checks:
            err = np.sum( np.abs((i-p) - np.transpose( np.dot(A, pix_n) )) )
            if err > 1e-6:
                raise RuntimeError('Error in computing where scattering vectors '
                                   'intersect with detector. Intersect not reproduced'
                                   ' (err: %f per pixel)' % (err / i.shape[0],) )

            if not np.sum(resid) < 1e-6:
                raise RuntimeError('Error in basis grid (residuals of point '
                                   'placement too large). Perhaps fast/slow '
                                   'vectors describing basis grid are linearly '
                                   'dependant?')

        pix_n = pix_n.T
        assert pix_n.shape[1] == 2 # s/f

        # see if the intersection in the plane is on the detector grid
        intersect = (pix_n[:,0] >= 0.0) * (pix_n[:,0] <= float(shape[0]-1)) *\
                    (pix_n[:,1] >= 0.0) * (pix_n[:,1] <= float(shape[1]-1))

        logger.debug('%.3f %% of pixels intersected by grid %d' % \
            ( (np.sum(intersect) / np.product(intersect.shape) * 100.0),
            grid_index) )

        return pix_n[intersect], intersect


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
        energy : float
            Energy of the beam (in keV)
        l : float
            The path length from the sample to the detector, in the same units
            as the detector dimensions.
        force_explicit : bool
            Forces the detector to be xyz_type explicit. Mostly for debugging.
            Recommend keeping `False`.

        Returns
        -------
        detector : odin.xray.Detector
            An instance of the detector that meets the specifications of the
            parameters
        """

        beam = Beam(photons_scattered_per_shot, energy=energy)

        if not force_explicit:

            p = np.array([-lim, -lim, l])   # corner position
            f = np.array([0.0, spacing, 0.0]) # slow scan is x
            s = np.array([spacing, 0.0, 0.0]) # fast scan is y

            dim = int(2*(lim / spacing) + 1)
            shape = (dim, dim)

            basis = BasisGrid()
            basis.add_grid(p, s, f, shape)

            detector = cls(basis, beam)

        else:
            x = np.arange(-lim, lim+spacing, spacing)
            xx, yy = np.meshgrid(x, x)

            xyz = np.zeros((len(x)**2, 3))
            xyz[:,0] = yy.flatten() # fast scan is y
            xyz[:,1] = xx.flatten() # slow scan is x
            xyz[:,2] = l

            detector = cls(xyz, beam)

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
        d = cls._from_serial(hdf['detector'])
        return d
    

class Shotset(object):
    """
    A collection of xray 'shots', and methods for anaylzing statistical
    properties across shots (each shot a single x-ray image).
    """

    def __init__(self, intensities, detector, mask=None):
        """
        Instantiate a Shotset class.

        Parameters
        ----------
        intensities : ndarray, float
            Either a list of one-D arrays, or a two-dimensional array. The first
            dimension should index shots, the second intensities for each pixel
            in that shot.

        detector : odin.xray.Detector
            A detector object, containing the pixel positions in space.

        Optional Parameters
        -------------------
        mask : ndarray, np.bool
            An array the same size (and shape -- 1d) as `intensities` with a
            'np.True' in all indices that should be kept, and 'np.False'
            for all indices that should be masked.
        """

        # parse detector
        if not isinstance(detector, Detector):
            raise ValueError('`detector` argument must be type: xray.Detector')
        else:
            self.detector = detector

        # parse intensities
        if type(intensities) == list:
            intensities = np.array(intensities)

        if type(intensities) == np.ndarray:
            s = intensities.shape

            if len(s) == 1:
                if not s[0] == self.detector.num_pixels:
                    raise ValueError('`intensities` does not have the same '
                                     'number of pixels as `detector`')
                self.intensities = intensities[None,:]

            elif len(s) == 2:
                if not s[1] == self.detector.num_pixels:
                    raise ValueError('`intensities` does not have the same '
                                     'number of pixels as `detector`')
                self.intensities = intensities

            else:
                raise ValueError('`intensities` has a invalid number of '
                                 'dimensions, must be 1 or 2 (got: %d)' % len(s))

        else:
            raise TypeError('`intensities` must be type ndarray')

        assert len(self.intensities.shape) == 2

        # parse mask
        if mask != None:
            mask = mask.flatten()
            if len(mask) != self.detector.num_pixels:
                raise ValueError('Mask must a len `detector.num_pixels` array')
            self.mask = np.array(mask.flatten()).astype(np.bool)
        else:
            #self.mask = np.zeros(self.detector.num_pixels, dtype=np.bool)
            self.mask = None


        return


    @property
    def num_shots(self):
        return self.intensities.shape[0]


    def __len__(self):
        return self.num_shots


    def __getitem__(self, key):
        """
        Slice the shotset into a smaller shotset.
        """
        if type(key) not in [np.ndarray, int]:
            raise TypeError('Only int or np.ndarray:dtype int can slice a Shot')
        new_i = self.intensities[key,:].copy()
        return Shotset(new_i, self.detector, self.mask)


    def __add__(self, other):
        if not isinstance(other, Shotset):
            raise TypeError('Cannot add types: %s and Shotset' % type(other))
        if not self.detector == other.detector:
            raise RuntimeError('shotset objects must share the same detector to add')
        if not np.all(self.mask == other.mask):
            raise RuntimeError('shotset objects must share the same mask to add')
        new_i = np.vstack(( self.intensities, other.intensities ))
        return Shotset(new_i, self.detector, self.mask)


    @property
    def average_intensity(self):
        return self.intensities.sum(0) # average over shots


    @staticmethod
    def num_phi_to_values(num_phi):
        """
        Converts `phi_spacing` to the explicit values, all in RADIANS.
        """
        phi_values = np.arange(0, 2.0*np.pi, 2.0*np.pi/float(num_phi))
        return phi_values


    @staticmethod
    def num_phi_to_spacing(num_phi):
        return 2.0*np.pi / float(num_phi)


    def assemble_image(self, shot_index=None, num_x=None, num_y=None):
        """
        Assembles the Shot object into a real-space image.

        Parameters
        ----------
        shot_index : int
            The shot inside the Shotset to assemble. If `None`, will assemble
            an average image.

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

        if shot_index == None:
            inten = self.average_intensity
        else:
            inten = self.intensities[shot_index,:]

        if (num_x == None) or (num_y == None):
            # todo : better performance if needed (implicit detector)
            num_x = len(self.detector.xyz[:,0])
            num_y = len(self.detector.xyz[:,1])

        points = self.detector.xyz[:,:2] # ignore z-comp. of detector

        x = np.linspace(points[:,0].min(), points[:,0].max(), num_x)
        y = np.linspace(points[:,1].min(), points[:,1].max(), num_y)
        grid_x, grid_y = np.meshgrid(x,y)

        grid_z = interpolate.griddata(points, inten,
                                      (grid_x,grid_y), method='nearest',
                                      fill_value=0.0)

        return grid_z


    def polar_grid(self, q_values, num_phi):
        """
        Return the pixels that comprise the polar grid in (q, phi) space.
        """

        phi_values = self.num_phi_to_values(num_phi)
        num_q = len(q_values)

        polar_grid = np.zeros((num_q * num_phi, 2))
        polar_grid[:,0] = np.repeat(q_values, num_phi)
        polar_grid[:,1] = np.tile(phi_values, num_q)

        return polar_grid


    def polar_grid_as_cart(self, q_values, num_phi):
        """
        Returns the pixels that comprise the polar grid in q-cartesian space,
        (q_x, q_y, q_z)
        """

        phi_values = self.num_phi_to_values(num_phi)
        num_q = len(q_values)

        pg_real = np.zeros((num_q * num_phi, 3))
        phis = np.tile(phi_values, num_q)
        pg_real[:,0] = np.repeat(q_values, num_phi) * np.cos( phis )
        pg_real[:,1] = np.repeat(q_values, num_phi) * np.sin( phis )

        return pg_real


    def interpolate_to_polar(self, q_values=None, num_phi=360, q_spacing=0.02):
        """
        Interpolate our cartesian-based measurements into a polar coordiante
        system.

        Parameters
        ----------
        q_values : ndarray OR list OF floats
            If supplied, the interpolation will only be performed at these
            values of |q|, and the `q_spacing` parameter will be ignored.

        num_phi : int
            The number of equally-spaced points around the azimuth to
            interpolate (e.g. `num_phi`=360 means points at 1 deg spacing).

        q_spacing : float
            The q-vector spacing, in inverse angstroms.

        Returns
        -------
        interpolated_intensities : ndarray, float
            The interpolated values. A three-D array, (shots, q_values, phis)

        polar_mask : ndarray, bool
            A mask of ones and zeros. Ones are kept, zeros masked. Shape and
            pixels correspond to `interpolated_intensities`.
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
            polar_intensities, polar_mask = self._explicit_interpolation(q_values, num_phi)
        elif self.detector.xyz_type == 'implicit':
            polar_intensities, polar_mask = self._implicit_interpolation(q_values, num_phi)
        else:
            raise RuntimeError('Invalid detector passed to Shot(), must be of '
                               'xyz_type {explicit, implicit}')

        return polar_intensities, polar_mask


    def _implicit_interpolation(self, q_values, num_phi):
        """
        Interpolate onto a polar grid from an `implicit` detector geometry.

        This detector geometry is specified by the x and y pixel spacing,
        the number of pixels in the x/y direction, and the top-left corner
        position.

        Notes
        -----
        --  The interpolation is performed in real space in basis vector (s/f)
            units
        --  The returned polar intensities are flattened, but each grid has data
            laid out as (q_values [slow], phi [fast])
        """

        # initialize output space for the polar data and mask
        num_q = len(q_values)
        polar_intensities = np.zeros((self.num_shots, num_q, num_phi))
        polar_mask        = np.zeros((num_q * num_phi), dtype=np.bool) # reshaped later
        q_vectors         = _q_grid_as_xyz(q_values, num_phi, self.detector.k)


        # --- loop over all the arrays that comprise the detector ---
        #     use grid interpolation on each

        int_start = 0 # start of intensity array correpsonding to `grid`
        int_end   = 0 # end of intensity array correpsonding to `grid`

        for g in range(self.detector._basis_grid.num_grids):

            p, s, f, size = self.detector._basis_grid.get_grid(g)

            # compute how many pixels this grid has
            n_int = int( np.product(size) )
            int_end += n_int

            # compute where the scattering vectors intersect the detector
            pix_n, intersect = self.detector._compute_intersections(q_vectors, g)

            if np.sum(intersect) == 0:
                logger.warning('Detector array (%d) had no pixels inside the \
                interpolation area!' % g)
                continue

            # --- loop over shots
            for i in range(self.num_shots):

                # interpolate onto the polar grid & update the inverse mask
                # --> perform the interpolation in pixel units, and then convert
                #     evaluated values to pixel units before evalutating

                # corner: (0,0); x/y size: 1.0; x is fast, y slow
                shot_pi = np.zeros(num_q * num_phi)
                shot_pm = np.zeros(num_q * num_phi, dtype=np.bool)

                interp = Bcinterp(self.intensities[i,int_start:int_end],
                                  1.0, 1.0, size[1], size[0], 0.0, 0.0)

                shot_pi[intersect] = interp.evaluate(pix_n[:,1], pix_n[:,0])
                polar_intensities[i,:,:] = shot_pi.reshape(num_q, num_phi)


            # mask points that missed
            polar_mask[intersect] = np.bool(True)

            # next, mask any points that should be masked by the real mask
            if self.mask == None:
                pass # if we have no work to do, skip this step...
            else:

                # to get the bicubic interpolation right, need to mask 16-px box
                # around any masked pixel. To do this, loop over masked px and
                # mask any polar pixel within 2-pixel units in either x or y dim

                assert self.mask.dtype == np.bool
                sub_mask = self.mask[int_start:int_end].reshape(size)
                sixteen_mask = filters.minimum_filter(sub_mask, size=(4,4),
                                                      mode='nearest')

                u = unique_rows( np.floor(pix_n) ).astype(np.int)
                polar_mask[intersect] = sixteen_mask[u] # false if masked

            # increment index for self.intensities -- the real/measured intst.
            int_start += n_int

        polar_mask = polar_mask.reshape(num_q, num_phi)

        return polar_intensities, polar_mask


    def _explicit_interpolation(self, q_values, num_phi):
        """
        Perform an interpolation to polar coordinates assuming that the detector
        pixels do not form a rectangular grid.

        NOTE: The interpolation is performed in polar momentum (q) space.
        """

        # initialize output space for the polar data and mask
        num_q = len(q_values)
        polar_intensities = np.zeros((self.num_shots, num_q, num_phi))
        polar_mask        = np.zeros(num_q * num_phi, dtype=np.bool)
        xy = self.detector.recpolar[:,[0,2]]

        # because we're using a "square" interplation method, wrap around one
        # set of polar coordinates to capture the periodic nature of polar coords
        add = ( xy[:,1] == xy[:,1].min() )
        xy_add = xy[add]
        xy_add[:,1] += 2.0 * np.pi
        aug_xy = np.concatenate(( xy, xy_add ))

        if self.mask != None:
            aug_mask = np.concatenate(( self.mask, self.mask[add] ))
        else:
            # slice all
            aug_mask = slice(0, self.detector.num_pixels + len(add))

        for i in range(self.num_shots):

            aug_int = np.concatenate(( self.intensities[i,:],
                                       self.intensities[i,add] ))

            # do the interpolation
            z_interp = interpolate.griddata( aug_xy[aug_mask], aug_int[aug_mask],
                                             self.polar_grid(q_values, num_phi),
                                             method='linear', fill_value=np.nan)

            polar_intensities[i,:,:] = z_interp.reshape(num_q, num_phi)

            # mask missing pixels (outside convex hull)
            nans = np.isnan(z_interp)
            polar_intensities[i,nans.reshape(num_q, num_phi)] = 0.0
            polar_mask[np.logical_not(nans)] = np.bool(True)

        polar_mask = polar_mask.reshape(num_q, num_phi)

        return polar_intensities, polar_mask


    def intensity_profile(self, q_spacing=0.05):
        """
        Averages over the azimuth phi to obtain an intensity profile.

        Optional Parameters
        -------------------
        q_spacing : float
            The resolution of the |q|-axis

        Returns
        -------
        intensity_profile : ndarray, float
            An n x 2 array, where the first dimension is the magnitude |q| and
            the second is the average intensity at that point < I(|q|) >_phi.
        """

        q = self.detector.recpolar[:,0]
        q_vals = np.arange(q_spacing, q.max(), q_spacing)

        ind = np.digitize(q, q_vals)

        avg = np.zeros(len(q_vals))
        for i in range(ind.max()):
            x = (ind == i)
            if np.sum(x) > 0:
                avg[i] = np.mean( self.average_intensity[x] )
            else:
                avg[i] = 0.0

        intensity_profile = np.vstack( (q_vals, avg) ).T

        return intensity_profile


    def intensity_maxima(self, smooth_strength=10.0):
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
        m = maxima( smooth(intensity[:,1], beta=smooth_strength) )
        return m


    @classmethod
    def simulate(cls, traj, detector, num_molecules, num_shots, traj_weights=None,
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

        num_shots : int
            The number of shots to simulate.

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
        shotset : odin.xray.Shotset
            A Shotset instance, containing the simulated shots.
        """

        I = np.zeros((num_shots, detector.num_pixels))

        for i in range(num_shots):
            I[i,:] = scatter.simulate_shot(traj, num_molecules, detector,
                                           traj_weights=traj_weights,
                                           finite_photon=finite_photon,
                                           force_no_gpu=force_no_gpu,
                                           device_id=device_id)

        ss = cls(I, detector)

        return ss


    def to_rings(self, q_values, num_phi=360):
        """
        Convert the shot to an xray.Rings object, for computing correlation
        functions and other properties in polar space.

        This automatically interpolates the dataset onto a polar grid and then
        converts those polar values into a class that facilitates computation
        in that space. See odin.xray.Rings for more info.

        Parameters
        ----------
        q_values : ndarray/list, float
            The values of |q| to extract rings at (in Ang^{-1}).

        num_phi : int
            The number of equally spaced points around the azimuth to
            interpolate onto (e.g. `num_phi`=360 means 1 deg spacing).
        """

        logger.info('Converting %d shots to polar space (Rings)' % self.num_shots)
        pi, pm = self.interpolate_to_polar(q_values=q_values, num_phi=num_phi)
        r = Rings(q_values, pi, self.detector.k, pm)

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

        shotdata = {}
        for i in range(self.num_shots):
            shotdata[('shot%d' % i)] = self.intensities[i,:]

        # if we don't have a mask, just save a single zero
        if self.mask == None:
            mask = np.array([0])
        else:
            mask = self.mask

        io.saveh(filename,
                 num_shots = np.array([self.num_shots]),
                 detector  = self.detector._to_serial(),
                 mask      = mask,
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


        # load from a shot file
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

            list_of_intensities = []
            d = Detector._from_serial(hdf['detector'])
            mask = hdf['mask']

            # check for our flag that there is no mask
            if np.all(mask == np.array([0])):
                mask = None

            for i in to_load:
                i = int(i)
                list_of_intensities.append( hdf[('shot%d' % i)] )

            hdf.close()


        elif filename.endswith('.cxi'):
            raise NotImplementedError() # todo

        else:
            raise ValueError('Must load a shotset file [.shot, .cxi]')


        return cls(list_of_intensities, d, mask)


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
            same shape as `polar_intensities`, but LESS THE FRIST DIMENSION.
            That is, the polar mask is the same for all shots. Can also be
            `None`, meaning no masked pixels
        """

        if not polar_intensities.shape[1] == len(q_values):
            raise ValueError('`polar_intensities` must have same len as '
                             '`q_values` in its second dimension.')

        if polar_mask == None:
            self.polar_mask = None
        elif type(polar_mask) == np.ndarray:
            if not polar_mask.shape == polar_intensities.shape[1:]:
                raise ValueError('`polar_mask` must have same shape as '
                                 '`polar_intensities[0,:,:]`,')
            if not polar_mask.dtype == np.bool:
                self.polar_mask = polar_mask.astype(np.bool)
            else:
                self.polar_mask = polar_mask
        else:
            raise TypeError('`polar_mask` must be np.ndarray or None')

        self._q_values         = np.array(q_values)           # q values of the ring data
        self.polar_intensities = np.copy(polar_intensities)   # copy data so don't over-write
        self.k                 = k                            # wave number

        return


    @property
    def num_shots(self):
        return self.polar_intensities.shape[0]


    @property
    def phi_values(self):
        return np.arange(0, 2.0*np.pi, 2.0*np.pi/float(self.num_phi))


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
    
        
    def _cospsi(self, q1, q2):
        """
        For each value if phi, compute the cosine of the angle between the
        reciprocal scattering vectors q1/q2 at angular separation phi.
        
        cos(psi) = f[phi, q1, q2]
        
        Parameters
        ----------
        q1/q2 : float
            The |q| values, in inv. ang.
            
        Returns
        -------
        cospsi : ndarray, float
            The cosine of psi, the angle between the scattering vectors.
        """
        
        # this function was formerly: get_cos_psi_vals
        
        t1     = np.pi/2. + np.arcsin( q1 / (2.*self.k) ) # theta 1 in spherical coor
        t2     = np.pi/2. + np.arcsin( q2 / (2.*self.k) ) # theta 2 in spherical coor
        cospsi = np.cos(t1)*np.cos(t2) + np.sin(t1)*np.sin(t2) *\
                 np.cos( self.phi_values )
              
        return cospsi
    

    def q_index(self, q, tolerance=1e-4):
        """
        Convert value of |q| (in inverse Angstroms) into the index used to
        slice `polar_intensities`.

        Parameters
        ----------
        q : float
            The value of |q| in inv Angstroms

        tolerance : float
            The tolerance in |q|. Will return values of q that are within this
            tolerance.

        Returns
        -------
        q_ind : int
            The index to slice `polar_intensities` at to get |q|.
        """
        
        # check if there are rings at q
        q_ind = np.where( np.abs(self.q_values - q) < tolerance )[0]
        
        if len(q_ind) == 0:
            raise ValueError("No ring data at q="+str(q) +" inv ang. " \
                             "There are only data for q="+", ".join(np.char.mod("%.2f",self.q_values) )  )
        elif len(q_ind) > 1:
            raise ValueError("Multiple q-values found! Try decreasing the value"
                             "of the `tolerance` parameter.")
        
        return int(q_ind)
    

    def depolarize(self, out_of_plane):
        """
        Applies a polarization correction to the rings.

        Parameters
        ----------
        out_of_plane : float
            fraction of polarization out of the synchrotron plane (between 0 and 1)
        """
        
        logger.warning("Warning, depolarize is UNTESTED!!")
        wave = self.k / 2. / np.pi
        
        for i in xrange(self.num_q):
            q = self.q_values[i]
            theta = np.arcsin( q*wave / 4./ np.pi)
            SinTheta = np.sin( 2 * theta )
            for j in xrange(self.num_shots):
                correction  = out_of_plane * ( 1. - SinTheta**2  * np.cos(self.phi_values)**2 )
                correction += (1.-out_of_plane) * (1. - SinTheta**2 * np.sin( self.phi_values)** 2)
                self.polar_intensities[j,i,:]  /= correction
                
        return
    

    def intensity_profile(self):
        """
        Averages over the azimuth phi to obtain an intensity profile.

        Returns
        -------
        intensity_profile : ndarray, float
            An n x 2 array, where the first dimension is the magnitude |q| and
            the second is the average intensity at that point < I(|q|) >_phi.
        """

        intensity_profile      = np.zeros( (self.num_q, 2), dtype=np.float )
        intensity_profile[:,0] = self._q_values.copy()

        # average over shots, phi
        if self.polar_mask != None:
            i = self.polar_intensities * self.polar_mask.astype(np.float)
        else:
            i = self.polar_intensities

        intensity_profile[:,1] = np.mean( np.mean(i, axis=2), axis=0)

        return intensity_profile


    def correlate_intra(self, q1, q2, num_shots=0, mean_only=False):
        """
        Does intRA-shot correlations for many shots.

        Parameters
        ----------
        q1 : float
            The |q| value of the first ring
        q2 : float
            The |q| value of the second ring

        Optional Parameters
        -------------------
        num_shots : int
            number of shots to compute correlators for
        mean_only : bool
            whether or not to return every correlation, or the average

        Returns
        -------
        intra : ndarray, float
            Either the average correlation, or every correlation as a 2d array
        """

        logger.debug("Correlating rings at %f / %f" % (q1, q2))

        q_ind1 = self.q_index(q1)
        q_ind2 = self.q_index(q2)

        if num_shots == 0: # then do correlation for all shots
            num_shots = self.num_shots

        rings1 = self.polar_intensities[:num_shots,q_ind1,:] # shots at ring1
        rings2 = self.polar_intensities[:num_shots,q_ind2,:] # shots at ring2

        # Check if mask exists
        if self.polar_mask != None:
            mask1 = self.polar_mask[q_ind1,:]
            mask2 = self.polar_mask[q_ind2,:]
        else:
            mask1 = None
            mask2 = None     

        return self._correlate_rows(rings1, rings2, mask1, mask2, mean_only)
    

    def correlate_inter(self, q1, q2, num_pairs=0, mean_only=False):
        """
        Does intER-shot correlations for many shots.

        Parameters
        ----------
        q1 : float
            The |q| value of the first ring
        q2 : float
            The |q| value of the second ring

        Optional Parameters
        -------------------
        num_pairs : int
            number of pairs of shots to compute correlators for
        mean_only : bool
            whether or not to return every correlation, or the average

        Returns
        -------
        inter : ndarray, float
            Either the average correlation, or every correlation as a 2d array
        """

        logger.debug("Correlating rings at %f / %f" % (q1, q2))

        q_ind1 = self.q_index(q1)
        q_ind2 = self.q_index(q2)

        max_pairs = self.num_shots * (self.num_shots - 1) / 2
        
        if (num_pairs == 0) or (num_pairs > max_pairs):
            inter_pairs = []
            for i in range(self.num_shots):
                for j in range(i+1, self.num_shots):
                    inter_pairs.append([i,j])
            inter_pairs = np.array(inter_pairs)
        else:
            inter_pairs = random_pairs(self.num_shots, num_pairs)

        rings1 = self.polar_intensities[inter_pairs[:,0],q_ind1,:] # shots at ring1
        rings2 = self.polar_intensities[inter_pairs[:,1],q_ind2,:] # shots at ring2

        # Check if mask exists
        if self.polar_mask != None:
            mask1 = self.polar_mask[q_ind1,:]
            mask2 = self.polar_mask[q_ind2,:]
        else:
            mask1 = None
            mask2 = None     

        return self._correlate_rows(rings1, rings2, mask1, mask2, mean_only)
        
        
    @staticmethod
    def _correlate_rows(x, y, x_mask=None, y_mask=None, mean_only=False):
        """
        Compute the circular correlation function across the rows of x,y. Note
        that *all* ODIN correlation functions are defined as:
        
                    C(x,y) = < (x - <x>) (y - <y>) > / <x><y>
        
        Parameters
        ----------
        x,y : np.ndarray, float
            2D arrays of size N x M, where N indexes "experiments" and M indexes
            an observation vector for each experiment.
            
        Optional Parameters
        -------------------
        x_mask,y_mask : np.ndarray, bool
            Arrays describing masks over the data. These are 1D arrays of size
            M, with a single value for each data point.
        
        mean_only : bool
            Return the mean of the correlation function. Default is to return
            each correlation individually.
            
        Returns
        -------
        corr : np.ndarray, float
            The N x M circular correlation function for each experiment. If
            `mean_only` is true, this is just a len-M array, averaged over
            the first dimension.
        """
        
        # do a shitload of typechecking -.-
        if len(x.shape) == 1:
            x = x[None,:]
        elif len(x.shape) > 2:
            raise ValueError('`x` must be two dimensional array')
            
        if len(y.shape) == 1:
            y = y[None,:]
        elif len(y.shape) > 2:
            raise ValueError('`y` must be two dimensional array')
            
        if not y.shape == x.shape:
            raise ValueError('`x`,`y` must have the same shape')
        
        n_row = x.shape[0]
        n_col = x.shape[1]

        if x_mask != None: 
            assert len(x_mask) == n_col
            x_mask = x_mask.astype(np.bool)

        if y_mask != None:
            assert len(y_mask) == n_col
            y_mask = y_mask.astype(np.bool)
            
            
        # --- actually compute some correlators
        # if no mask
        if ((x_mask == None) and (y_mask == None)):
            
            x_bar = x.mean(axis=1)[:,None]
            y_bar = y.mean(axis=1)[:,None]
            
            # use d-FFT + convolution thm
            ffx = fftpack.fft(x - x_bar, axis=1)
            ffy = fftpack.fft(y - y_bar, axis=1)
            corr = np.real(fftpack.ifft( ffx * np.conjugate(ffy), axis=1 ))
            assert corr.shape == (n_row, n_col)
            
            # normalize
            corr = corr / ( float(n_col) * x_bar * y_bar )
                    
        # if using mask
        else:
            corr = np.zeros((n_row, n_col))
            for i in range(n_row):
                corr[i,:] = gap_correlate(x[i,:] * x_mask[:], y[i,:] * y_mask[:])

        if mean_only:
            corr = corr.mean(axis=0) # average all shots

        return corr
    

    def _convert_to_kam(self, q1, q2, corr):
        """
        Corrects the azimuthal intensity correlation on a detector for detector 
        curvature.
        Considers the Friedel Pairs C (cos(psi) ) = C( cos( -psi) )

        Parameters
        ----------
        q1,q2 : float, float
            Inverse angstroms values of the intenisty rings
        corr : ndarray,  float
            Azimuathl correlation function of intensities along ring in q space

        Returns
        -------
        kam_corr : ndarray, float
            The Kam correlation function and the cos(psi) values
        """
        
        if not len(corr.shape) == 1:
            raise ValueError('`corr` must be a one-dimensional array')
        
        cosPsi  = self._cospsi(q1,q2)           # azimuathal to cos(psi)
        cosPsi  = np.append( cosPsi, -cosPsi )  # Adding the Friedel pairs...
        newCor  = np.append( corr, corr )       # C [cos(psi) ] = C [cos(-psi)]
        
        kam_corr = np.vstack((cosPsi, newCor)).T
        kam_corr = kam_corr[ np.argsort(kam_corr[:,0]) ] # sort ascending angle
        
        return kam_corr


    def legendre(self, q1, q2, order, use_inter_statistics=False):
        """
        Project the correlation functions onto a set of legendre polynomials,
        and return the coefficients of that projection.

        Parameters
        ----------
        order : int
            The order at which to truncate the polynomial expansion. Note that
            this function projects only onto even numbered Legendre polynomials.

        Optional Parameters
        -------------------
        use_inter_statistics : bool
            Whether or not to subtract inter-shot statistics from the
            correlation function before projecting it. This can help remove
            detector artifacts at the cost of a small computational overhead.

        Returns
        -------
        c: np.ndarray, float
            An array of the legendre coefficients. Contains all coefficients
            (both even and odd) up to order `order`
        """

        # iterate over each pair of rings in the collection and project the
        # correlation between those two rings into the Legendre basis

        if use_inter_statistics:
            corr = self.correlate_intra(q1, q2, mean_only=True) - \
                   self.correlate_inter(q1, q2, mean_only=True)
        else:
            corr = self.correlate_intra(q1, q2, mean_only=True)
        
        corr = self._convert_to_kam( q1, q2, corr )

        # tests indicate this is a good numerical projection
        c = np.polynomial.legendre.legfit(corr[:,0], corr[:,1], order-1)
        return c


    def legendre_matrix(self, order, use_inter_statistics=False):
        """
        Project the correlation functions onto a set of legendre polynomials,
        and return the coefficients of that projection.

        Parameters
        ----------
        order : int
            The order at which to truncate the polynomial expansion. Note that
            this function projects only onto even numbered Legendre polynomials.

        Optional Parameters
        -------------------
        use_inter_statistics : bool
            Whether or not to subtract inter-shot statistics from the
            correlation function before projecting it. This can help remove
            detector artifacts at the cost of a small computational overhead.

        Returns
        -------
        Cl: np.ndarray, float
            An array of the legendre coefficients. Contains all coefficients
            (both even and odd) up to order `order`. The returned object is a
            3-D array indexed by

                (order, q_ind1, q_ind2)

            where the q_ind values are the indices that map onto self.q_values.
        """

        # initialize space for coefficients
        Cl = np.zeros( (order, self.num_q, self.num_q) )

        # iterate over each pair of rings in the collection and project the
        # correlation between those two rings into the Legendre basis

        for i in range(self.num_q):
            q1 = self.q_values[i]
            for j in range(i,self.num_q):
                q2 = self.q_values[j]
                c  = self.legendre(q1, q2, order=order, 
                                   use_inter_statistics=use_inter_statistics)
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

        num_molecules : int
            The number of molecules estimated to be in the `beam`'s focus.

        q_values : ndarray/list, float
            The values of |q| to extract rings at (in Ang^{-1}).

        num_phi : int
            The number of equally spaced points around the azimuth to
            interpolate onto (e.g. `num_phi`=360 means 1 deg spacing).

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
        q_values = np.array(q_values)

        qxyz = _q_grid_as_xyz(q_values, num_phi, k)

        # --- simulate the intensities ---

        polar_intensities = np.zeros((num_shots, len(q_values), num_phi))

        for i in range(num_shots):
            I = scatter.simulate_shot(traj, num_molecules, qxyz,
                                      traj_weights=traj_weights,
                                      finite_photon=finite_photon,
                                      force_no_gpu=force_no_gpu,
                                      device_id=device_id)
            polar_intensities[i,:,:] = I.reshape(len(q_values), num_phi)

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


def _q_grid_as_xyz(q_values, num_phi, k):
    """
    Generate a q-grid in cartesian space: (q_x, q_y, q_z).

    Parameters
    ----------
    q_values : ndarray/list, float
        The values of |q| to extract rings at (in Ang^{-1}).

    num_phi : int
        The number of equally spaced points around the azimuth to
        interpolate onto (e.g. `num_phi`=360 means 1 deg spacing).

    Returns
    -------
    qxyz : ndarray, float
        An N x 3 array of (q_x, q_y, q_z)
    """

    phi_values = np.linspace( 0.0, 2.0*np.pi, num=num_phi )
    num_q = len(q_values)

    # q_h is the magnitude projection of the scattering vector on (x,y)
    q_z = - np.power(q_values, 2) / (2.0 * k)
    q_h = q_values * np.sqrt( 1.0 - np.power( q_values / (2.0 * k), 2 ) )

    # construct the polar grid:
    qxyz = np.zeros(( num_q * num_phi, 3 ))
    qxyz[:,0] = np.repeat(q_h, num_phi) * np.cos(np.tile(phi_values, num_q)) # q_x
    qxyz[:,1] = np.repeat(q_h, num_phi) * np.sin(np.tile(phi_values, num_q)) # q_y
    qxyz[:,2] = np.repeat(q_z, num_phi)                                      # q_z

    return qxyz
