# THIS FILE IS PART OF ODIN


"""
Classes, methods, functions for use with xray scattering experiments.
"""

# ---------------- #
IGNORE_NAN = True  #
# ---------------- #

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

import numpy as np
from scipy import interpolate, fftpack
from bisect import bisect_left

from odin import utils
from odin.data import cromer_mann_params

from mdtraj import trajectory, io

# try to import the gpuscatter module
GPU = True
try:
    import gpuscatter
except ImportError as e:
    logger.warning('Could not find `gpuscatter` module, proceeding without it.'
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
    
    def __init__(self, xyz, path_length, k, xyz_type='explicit'):
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
        """
        
        if xyz_type == 'explicit':
            
            if type(xyz) != np.ndarray:
                raise ValueError('Explicit xyz pixels required, `xyz` must be np.ndarray')
            
            self.pixels = xyz
            self.grid_list = None
            
            self.num_q = xyz.shape[0]
            self._xyz_type = 'explicit'
                
        elif xyz_type == 'implicit':
            
            # I am being a little tricky here -- by passing the grid_list into
            # the xyz argument. The safety of this move is predicated on
            # the user employing the xyz_type kwarg appropriately. Hopefully 
            # this doesn't cause too much trouble later... -- TJL
            if type(xyz) != list:
                raise ValueError('Implicit xyz pixels required, `xyz` must be list')
            self._xyz_type = 'implicit'
            
            self.grid_list = xyz
            self.pixels = None
            
        else:
            raise ValueError("`xyz_type` must be one of {'explicit', 'implicit'}")
        
        # parse wavenumber
        if isinstance(k, Beam):
            self.k = k.wavenumber
        elif type(k) in [float, np.float64, np.float32]:
            self.k = k
        else:
            raise ValueError('`k` must be a float or odin.xray.Beam')
        
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
        d = Detector(grid_list, path_length, k, xyz_type='implicit')
        return d
      
        
    def implicit_to_explicit(self):
        """
        Convert an implicit detector to an explicit one (where the xyz pixels
        are stored in memory).
        """
        if not self.xyz_type == 'implicit':
            raise Exception('Detector must have xyz_type implicit for conversion.')
        self.pixels = self.xyz
        self._xyz_type = 'explicit'
        return
        
             
    @property
    def xyz_type(self):
        return self._xyz_type
        
        
    @property
    def xyz(self):
        if self.xyz_type == 'explicit':
            return self.pixels
        elif self.xyz_type == 'implicit':
            return np.concatenate( [self._grid_from_implicit(*t) for t in self.grid_list] )
        
        
    def _grid_from_implicit(self, basis, size, corner):
        """
        From an x, y, z basis vector set, and the dimensions of the grid,
        generates an explicit xyz grid.
        """
        
        assert self.xyz_type == 'implicit'
        
        if basis[0] != 0.0:
            x_pixels = np.arange(0, basis[0]*size[0], basis[0])
        else:
            x_pixels = np.zeros( size[0] )
        
        if basis[1] != 0.0:
            y_pixels = np.arange(0, basis[1]*size[1], basis[1])
        else:
            y_pixels = np.zeros( size[1] )
        
        if basis[2] != 0.0:
            z_pixels = np.arange(0, basis[2]*size[2], basis[2])
        else:
            z_pixels = np.zeros( size[2] )
        
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
        return self._real_to_recpolar(self.real)
        
        
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
        
        # generate unit vectors in the pixel direction, origin at sample
        S = self.real.copy()
        S = self._unit_vector(S)
        
        # generate unit vectors in the z-direction
        S0 = np.zeros(xyz.shape)
        S0[:,2] = np.ones(xyz.shape[0])
        
        q = self.k * (S - S0)
        
        return q
        
        
    def _real_to_recpolar(self, xyz):
        """
        Convert the real-space to reciprocal-space in polar form, that is
        (|q|, theta , phi).
        """
        reciprocal_polar = self._to_polar( self._real_to_reciprocal(xyz) )
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
        polar[:,2] = utils.arctan3(vector[:,1], vector[:,0])     # y first!
        
        return polar
        
        
    @classmethod
    def generic(cls, spacing=2.0, lim=100.0, energy=10.0, flux=100.0, l=50.0, 
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
            dim = 2*(lim / spacing) + 1
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
    def generic_polar(cls, q_spacing=0.02, q_lim=5.0, angle_spacing=1.0,
                      energy=10.0, flux=100.0, l=50.0):
        """
        Generates a simple grid detector that can be used for testing
        (factory function). 

        Optional Parameters
        -------------------
        q_spacing : float
            The |q| grid spacing
        q_lim : float
            The upper limit of the q-vector
        angle_spacing : float
            The angular grid spacing, in degrees
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
        # todo : implement
        
        raise NotImplementedError()
        # beam = Beam(flux, energy=energy)
        # 
        # x = np.arange(-lim, lim+spacing, spacing)
        # xx, yy = np.meshgrid(x, x)
        # 
        # xyz = np.zeros((len(x)**2, 3))
        # xyz[:,0] = xx.flatten()
        # xyz[:,1] = yy.flatten()
        # 
        # detector = Detector(xyz, l, beam)

        # return detector
        
        
    def save(self, filename):
        """
        Writes the current Detector to disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file to save.
        """
        
        # todo : extend to implicit detectors
        
        if filename[-4:] != '.dtc':
            filename += '.dtc'
        
        io.saveh(filename, dxyz=self.xyz, dpath_length=np.array([self.path_length]), 
                 dk=np.array([self.k]))
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
        
        # todo : extend to implicit detectors
        
        if not filename.endswith('.dtc'):
            raise ValueError('Must load a detector file (.dtc extension)')
        
        hdf = io.loadh(filename)
        
        xyz = hdf['dxyz']
        path_length = hdf['dpath_length'][0]
        k = float(hdf['dk'][0])
        
        return Detector(xyz, path_length, k)
        
        
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
                
        self.intensities = intensities
        self.detector = detector
        self.interpolate_to_polar()
        
        
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
            An n x 3 array, where n in the total number of data points.
                self.polar_intensities[:,0] -- magnitude of q, |q|
                self.polar_intensities[:,1] -- azmuthal angle phi
                self.polar_intensities[:,2] -- intensity I(|q|, phi)
        """
        
        # construct a polar grid from the parameters passed above, and store
        # a lot of stuff in `self`
        self.q_spacing = q_spacing
        self.phi_spacing = phi_spacing * (2.0*np.pi/360.) # conv. to radians
        self._generate_polar_grid()
        self.polar_mask = np.zeros(self.polar_intensities.shape[0], dtype=np.bool)
        
        if self.detector.xyz_type == 'explicit':
            # todo : check for rectangular grid and then use structured
            self._unstructured_interpolation()
        elif self.detector.xyz_type == 'implicit':
            self._implicit_interpolation()
        else:
            raise RuntimeError('Invalid detector passed to Shot()')
        
        
    def _generate_polar_grid(self):
        
        # determine the bounds of the grid, and the discrete points to use
        self.phi_values = np.arange(0.0, 2.0*np.pi, self.phi_spacing)
        self.num_phi = len(self.phi_values)
        
        self.q_min = np.min( self.detector.recpolar[:,0] )
        self.q_max = np.max( self.detector.recpolar[:,0] )
        self.q_values = np.arange(self.q_min, self.q_max+self.q_spacing, self.q_spacing)
        self.num_q = len(self.q_values)
        
        self.num_datapoints = self.num_phi * self.num_q
        
        # generate the polar grid to interpolate onto
        self.polar_intensities = np.zeros((self.num_datapoints, 3))
        self.polar_intensities[:,0] = np.tile(self.q_values, self.num_phi)
        self.polar_intensities[:,1] = np.repeat(self.phi_values, self.num_q)
        
        return
        
        
    def _implicit_interpolation(self):
        """
        Interpolate onto a polar grid from an `implicit` detector geometry.
        """
        
        # loop over all the arrays that comprise the detector and use
        # grid interpolation on each
        int_start = 0 # start of intensity array correpsonding to `grid`
        int_end   = 0 # end of intensity array correpsonding to `grid`
        
        # generate a mask that is true where we successfully interpolated
        inv_mask = np.zeros(self.polar_intensities.shape[0], dtype=np.bool)
        
        # convert the polar to cartesian coords for comparison to detector
        xp = self.polar_intensities[:,0] * np.cos( self.polar_intensities[:,1] )
        yp = self.polar_intensities[:,0] * np.sin( self.polar_intensities[:,1] )
        
        for grid in self.detector.grid_list:
            
            basis, size, corner = grid
            n_int = size[0] * size[1] * size[2]
            int_end += n_int
            
            if basis[0] != 0.0:
                x_pixels = np.arange(0, basis[0]*size[0], basis[0])
            else:
                x_pixels = np.zeros( size[0] )

            if basis[1] != 0.0:
                y_pixels = np.arange(0, basis[1]*size[1], basis[1])
            else:
                y_pixels = np.zeros( size[1] )

            if basis[2] != 0.0:
                z_pixels = np.arange(0, basis[2]*size[2], basis[2])
            else:
                z_pixels = np.zeros( size[2] )
            
            i = self.intensities[int_start:int_end].reshape(size[0], size[1])
            int_start += n_int
            f = interpolate.RectBivariateSpline(x_pixels, y_pixels, i)
            
            # find the indices of the polar grid that are inside the boundaries
            # of the current detector array we're interpolating over
            # todo : this is probably a slow way to go...
            p_ind_x = np.intersect1d( np.where( xp < x_pixels[0] )[0], 
                                      np.where( xp > x_pixels[-1] )[0] )
            p_ind_y = np.intersect1d( np.where( yp < y_pixels[0] )[0], 
                                      np.where( yp > y_pixels[-1] )[0] )                                    
            p_inds = np.intersect1d(p_ind_x, p_ind_y)
            
            if len(p_inds) == 0:
                logger.warning('Detector array had no overlapping polar pixels!')
                continue
            
            # interpolate onto the polar grid & update the inverse mask
            self.polar_intensities[p_inds,2] = f.ev(xp[p_inds], yp[p_inds])
            inv_mask[p_ind] = np.bool(True)
            
        self.polar_mask += np.bool(True) - inv_mask
        self._update_mask()
        
        return
        
        
    def _structured_interpolation(self):
        # todo
        # self._update_mask()
        raise NotImplementedError()
        
        
    def _unstructured_interpolation(self):
        """
        Perform an interpolation to polar coordinates assuming that the detector
        pixels do not form a rectangular grid.
        """
        
        # interpolate onto polar grid : delauny triangulation + nearest neighbour
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
        z_interp = interpolate.griddata( aug_xy, aug_int, self.polar_intensities[:,:2],
                                         method='linear', fill_value=np.nan )

        self.polar_intensities[:,2] = z_interp.flatten()

        # mask missing pixels (outside convex hull)
        nan_ind = np.where( np.isnan(z_interp) )[0]
        self.polar_intensities[nan_ind,2] = 0.0
        #self.polar_mask[nan_ind] = np.bool(True)
        
        self._update_mask()
        
        return
        
        
    def _update_mask(self):
        self.polar_intensities[:,2] = np.ma.array(self.polar_intensities[:,2],
                                                  mask=self.polar_mask)
        
    def unmask_all(self):
        """
        Reveals all masked coordinates.
        """
        self.masked_real_pixels = []
        self.masked_polar_pixels = []
        self.polar_intensities = np.array(self.polar_intensities) # unmask
        
        
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
            such that self.intensities[index,2] -> I(q,phi)
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
        return np.array(self.polar_intensities)[self._intensity_index(q,phi),2]
        
        
    def I_ring(self, q):
        """
        Return the intensities around a ring at `q`
        """
        q = self._nearest_q(q)
        ind = self._q_index(q)
        return np.array(self.polar_intensities)[ind,2]
        
        
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
        intensity = (self.polar_intensities[ind,2]).mean()
        
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

        intensity_profile = np.array(intensity_profile) # rm mask
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
        a = utils.smooth(intensity[:,1], beta=smooth_strength)
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
        
        x = self.I_ring(q1)
        y = self.I_ring(q2)
        y = np.roll(y, i)
        
        # this should work with masking
        corr = ( (x-x.mean()) * (y-y.mean()) ).mean() / (x.std() * y.std())
        
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
        
        # As dermen has pointed out, here we need to be a little careful with
        # how we deal with masked pixels. If we subtract the means from the 
        # pixels first, then perform the correlation, we can safely ignore
        # masked pixels so long as (1) we make sure their value is zero, and
        # (2) we don't include them in the count `n_delta` of total corration
        # pairs.
        
        q1 = self._nearest_q(q1)
        q2 = self._nearest_q(q2)
        
        x = self.I_ring(q1)
        if np.abs(q1 - q2) < 1e-6:
            y = x.copy()
        else:
            y = self.I_ring(q2)
        assert len(x) == len(y)
        n_theta = len(x)
        
        logger.debug("Correlating rings at %f / %f" % (q1, q2))
        
        xstd = x.std()
        ystd = y.std()
        
        x -= x.mean()
        y -= y.mean()
        
        # todo : ensure a zero gets plugged in at all mask positions
        
        correlation_ring = np.zeros((n_theta, 2))
        correlation_ring[:,0] = self.phi_values
        
        # use d-FFT + convolution thm
        ffx = fftpack.fft(x)
        ffy = fftpack.fft(y)
        iff = np.real(fftpack.ifft( ffx * ffy ))
        
        correlation_ring[:,1] = iff # / (np.linalg.norm(x) * np.linalg.norm(y)) # (xstd * ystd)
        
        m = correlation_ring[:,1].max()
        for i in range(correlation_ring.shape[0]):
            correlation_ring[i,1] = correlation_ring[i,1] / m
        
        return correlation_ring
        
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
    
        traj_weights : ndarray, float
            If `traj` contains many structures, an array that provides the Boltzmann
            weight of each structure. Default: if traj_weights == None, weights
            each structure equally.
    
        force_no_gpu : bool
            Run the (slow) CPU version of this function.
    
        Returns
        -------
        shot : odin.xray.Shot
            A shot instance, containing the simulated shot.
        """
        I = simulate_shot(traj, num_molecules, detector, traj_weights, 
                          force_no_gpu, device_id=device_id)
        shot = Shot(I, detector)
        return shot
        
        
    def save(self, filename):
        """
        Writes the current Shot data to disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file to save.
        """
        
        if not filename.endswith('.shot'):
            filename += '.shot'
        
        shotdata = {'shot1' : self.intensities}
        
        io.saveh(filename, 
                 num_shots    = np.array([1]), # just one shot :)
                 dxyz         = self.detector.xyz,
                 dpath_length = np.array([self.detector.path_length]), 
                 dk           = np.array([self.detector.k]),
                 **shotdata)
                 
        logger.info('Wrote %s to disk.' % filename)


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
        if not filename.endswith('.shot'):
            raise ValueError('Must load a detector file (.shot extension)')
        
        hdf = io.loadh(filename)
        
        num_shots   = hdf['num_shots'][0]
        xyz         = hdf['dxyz']
        path_length = hdf['dpath_length'][0]
        k           = hdf['dk'][0]
        intensities = hdf['shot1']
        
        if num_shots != 1:
            logger.warning('You loaded a .shot file that contains multiple shots'
                           ' into a single Shot instance... taking only the'
                           ' first shot of the set (look into Shotset.load()).')
        
        d = Detector(xyz, path_length, k)
        
        return Shot(intensities, d)


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
        
        
    def _check_qvectors_same(self, epsilon=1e-6):
        """
        For a lot of the below to work, we need to make sure that all of the 
        q-phi grids in the shots are the same. If some are different, here
        we recalculate them.
        """
        q_phis = self.shots[0].polar_intensities[:,:2]
        for shot in self.shots:
            diff = np.sum(np.abs(shot.polar_intensities[:,:2] - q_phis))
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
    
        shotlist = []
        for i in range(num_shots):
            I = simulate_shot(traj, num_molecules, detector, traj_weights, force_no_gpu, device_id=device_id)
            shot = Shot(I, detector)
            shotlist.append(shot)
        
        return Shotset(shotlist)
        
        
    def save(self, filename):
        """
        Writes the current Shot data to disk.

        Parameters
        ----------
        filename : str
            The path to the shotset file to save.
        """

        if not filename.endswith('.shot'):
            filename += '.shot'

        shotdata = {}
        for i in range(self.num_shots):
            shotdata[('shot%d' % i)] = self.shots[i].intensities

        io.saveh(filename, 
                 num_shots    = np.array([self.num_shots]),
                 dxyz         = self.shots[0].detector.xyz,
                 dpath_length = np.array([self.shots[0].detector.path_length]), 
                 dk           = np.array([self.shots[0].detector.k]),
                 **shotdata)

        logger.info('Wrote %s to disk.' % filename)


    @classmethod
    def load(cls, filename):
        """
        Loads the a Shot from disk. Must be `.shot` or `.cxi` format.

        Parameters
        ----------
        filename : str
            The path to the shotset file.

        Returns
        -------
        shotset : odin.xray.Shotset
            A shotset object
        """

        if filename.endswith('.shot'):
            hdf = io.loadh(filename)

            num_shots   = hdf['num_shots'][0]
            xyz         = hdf['dxyz']
            path_length = hdf['dpath_length'][0]
            k           = hdf['dk'][0]
        
            d = Detector(xyz, path_length, k)
        
            list_of_shots = []
            for i in range(num_shots):
                list_of_shots.append( Shot( hdf[('shot%d' % i)], d ) )
            
            
        elif filename.endswith('.cxi'):
            raise NotImplementedError() # todo
            
        else:
            raise ValueError('Must load a shotset file [.shot, .cxi]')

        return Shotset(list_of_shots)

    
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

    if traj_weights == None:
        traj_weights = np.ones( traj.n_frames )
    traj_weights /= traj_weights.sum()
        
    num_per_shapshot = np.random.multinomial(num_molecules, traj_weights, size=1)

    # get detector
    qx = detector.reciprocal[:,0].astype(np.float32)
    qy = detector.reciprocal[:,1].astype(np.float32)
    qz = detector.reciprocal[:,2].astype(np.float32)
    num_q = len(qx)
    assert( detector.num_q == num_q )

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
    intensities = np.zeros(detector.num_q, dtype=np.float64) # should be double
    
    for i,num in enumerate(num_per_shapshot):
        if int(num) > 0:
    
            num = int(num)

            # pull xyz coords
            rx = traj.xyz[i,:,0].flatten().astype(np.float32)
            ry = traj.xyz[i,:,1].flatten().astype(np.float32)
            rz = traj.xyz[i,:,2].flatten().astype(np.float32)

            # choose the number of molecules (must be multiple of 512)
            num = num - (num % 512)
            bpg = num / 512

            # generate random numbers for the rotations in python (much easier)
            rand1 = np.random.rand(num).astype(np.float32)
            rand2 = np.random.rand(num).astype(np.float32)
            rand3 = np.random.rand(num).astype(np.float32)

            # run dat shit
            if force_no_gpu:
                logger.debug('Running CPU computation')
                raise NotImplementedError('')

            else:
                logger.debug('Sending calculation to GPU device...')
                device_id = int(0)
                bpg = int(bpg)
                out_obj = gpuscatter.GPUScatter(device_id,
                                                bpg, qx, qy, qz,
                                                rx, ry, rz, aid,
                                                cromermann,
                                                rand1, rand2, rand3, num_q)
                logger.info('Retrived data from GPU.')
                assert( len(out_obj.this[1]) == num_q )
                intensities += out_obj.this[1].astype(np.float64)
    
    # check for NaNs in output
    if np.isnan(np.sum(intensities)):
        ind = np.where(np.isnan(intensities) == True)
        n_nan = len(ind)
        logger.critical('%d NaNs in output...' % n_nan)
        if IGNORE_NAN:
            intensities[ind] = 0.0
            logger.critical('Set NaNs to zero... be careful!')
            assert not np.isnan(np.sum(intensities))
        else:
            raise RuntimeError('Fatal error, NaNs detected in scattering output!')
    
    # check for negative values in output
    if len(intensities[intensities < 0.0]) != 0:
        ind = np.where(intensities < 0.0)[0]
        n_neg = len(ind)
        logger.critical('%d negative intensities in output..' % n_neg)
        if IGNORE_NAN:
            intensities[ind] = 0.0
            logger.critical('Set negative values to zero... be careful!')
            assert( len(intensities[intensities < 0.0]) == 0 )
        else:
            raise RuntimeError('Fatal error, negative intensities detected in scattering output!')
    
    return intensities
