
"""
Various parsers:
-- CBF        (crystallographic binary format)
-- Kitty H5   (CXI pyana spinoff)
-- CXI        (coherent xray imaging format)
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

import inspect
import tables
import re
import hashlib
import yaml
import tables
import fabio
from base64 import b64encode

import numpy as np

from odin import xray
from odin.math2 import CircularHough


class CBF(object):
    """
    A class for parsing CBF files. Depends on fabio.
    """
    
    def __init__(self, filename):
        """
        A light handle on a CBF file.
        
        Parameters
        ----------
        filename : str
            The path to the CBF file.
        """
        
        logger.info('Reading: %s' % filename)
        self.filename = filename
        
        # extract all interesting stuff w/fabio
        self._fabio_handle = fabio.open(filename)
        self._info = self._fabio_handle.header
        self._parse_array_header( self._info['_array_data.header_contents'] )
        self.intensity_dtype = self._convert_dtype(self._info['X-Binary-Element-Type'])
            
        logger.debug('Finished loading file')
        
        
    @property
    def md5(self):
        return self._info['Content-MD5']
        
    @property
    def intensities_shape(self):
        shp = (int(self._info['X-Binary-Size-Second-Dimension']), 
               int(self._info['X-Binary-Size-Fastest-Dimension']))
        return shp
        
    @property
    def pixel_size(self):
        p = self._info['Pixel_size'].split()
        assert p[1].strip() == p[4].strip()
        assert p[2].strip() == 'x'
        return (float(p[0]), float(p[3]))
        
    @property
    def path_length(self):
        # assume units are the same for all dims
        d, unit = self._info['Detector_distance'].split() 
        return float(d)
        
    @property
    def wavelength(self):
        return float(self._info['Wavelength'].split()[0])
        
    @property
    def polarization(self):
        return float(self._info['Polarization'])
        
    @property
    def center(self):
        """
        The indicies of the pixel nearest the center
        """
        if not hasattr(self, 'center'):
            self._center = self._find_center()
        return self._center
        
    @property
    def corner(self):
        """
        The bottom left corner position, in real space.
        """
        #return (self.pixel_size[0] * self.center[0], self.pixel_size[1] * self.center[1])
        return (0.0, 0.0)
        
    @property
    def intensities(self):
        return self._fabio_handle.data


    def _convert_dtype(self, dtype_str):
        """
        Converts `dtype_str`, straight from the cbf file, to the right numpy
        dtype
        """
        
        # TJL: I'm just guessing the names for most of these....
        # the cbflib docs are useless!!
        conversions = {"signed 32-bit integer" : np.int32,
                       "unsigned 32-bit integer" : np.uint32,
                       "32-bit float" : np.float32,
                       "64-bit float" : np.float64}
        
        try:
            dtype = conversions[dtype_str]
        except KeyError as e:
            raise ValueError('Binary-Element-Type: %s has no know numpy '
                             'counterpart. Contact the dev team if you believe '
                             'this is wrong -- it may be an unexpected string.'
                             % dtype_str)
        
        return dtype
    
        
    def _parse_array_header(self, array_header):
        """
        Fabio provides an '_array_data.header_contents' key entry in that
        needs to be parsed. E.g. for a test PILATUS detector file generated at
        SSRL, this dictionary entry looks like
        
        fabio_object.header['_array_data.header_contents'] = 
        '# Detector: PILATUS 6M, S/N 60-0101 SSRL\r\n# 2012/Apr/09 20:02:10.800
        \r\n# Pixel_size 172e-6 m x 172e-6 m\r\n# Silicon sensor, thickness 0.00
        0320 m\r\n# Exposure_time 9.997700 s\r\n# Exposure_period 10.000000 s\r\
        n# Tau = 110.0e-09 s\r\n# Count_cutoff 1060885 counts\r\n# Threshold_set
        ting 6000 eV\r\n# N_excluded_pixels = 1685\r\n# Excluded_pixels: badpix_
        mask.tif\r\n# Flat_field: (nil)\r\n# Trim_directory: p6m0101_T8p0_vrf_m0
        p3_090729\r\n# Wavelength 0.7293 A\r\n# Energy_range (0, 0) eV\r\n# Dete
        ctor_distance 0.20000 m\r\n# Detector_Voffset 0.00000 m\r\n# Beam_xy (12
        31.50, 1263.50) pixels\r\n# Flux 0.0000 ph/s\r\n# Filter_transmission 1.
        0000\r\n# Start_angle 90.0000 deg.\r\n# Angle_increment 0.0100 deg.\r\n#
         Detector_2theta 0.0000 deg.\r\n# Polarization 0.990\r\n# Alpha 0.0000 d
        eg.\r\n# Kappa 0.0000 deg.\r\n# Phi 90.0000 deg.\r\n# Chi 0.0000 deg.\r
        \n# Oscillation_axis X, CW\r\n# N_oscillations 1'
        
        This function makes some sense of this mess.
        
        Parameters
        ----------
        array_header : str
            Something that looks like the dictionary value above.
        
        This function injects this information into self._info.
        """

        logger.debug('Reading header info...')

        items = array_header.split('#')
        
        for item in items:
            split = item.strip().split(' ')
            if len(split) > 1:
                k = split[0].strip().lstrip(':')
                self._info[k] = ' '.join(split[1:]).strip().lstrip('=')
        
        return
    

    def _check_md5(self):
        """
        Check the data are intact by computing the md5 checksum of the binary
        data, and comparing it to an analagous md5 computed when the file was
        generated.
        """
        
        # This is a cute idea but I have no idea what data the md5 is performed
        # on, or how to retrieve that data from the file. This function is
        # currently broken (close to working)
        
        md5 = hashlib.md5()
        md5.update(self.intensities.flatten().tostring()) # need to feed correct data in here...
        data_md5 = b64encode(md5.digest())
        if not md5.hexdigest() == self.md5:
            logger.critical("Data MD5:    %s" % data_md5)
            logger.critical("Header MD5:  %s" % self.md5)
            raise RuntimeError('Data and stored md5 hashes do not match! Data corrupted.')
    
            
    def _find_center(self):
        """
        Find the center of any Bragg rings (aka the location of the x-ray beam).
        
        Returns
        -------
        center : tuple of ints
            The indicies of the pixel nearest the center of the Bragg peaks.
        """
        #CM = CircularHough(radii=10, threshold=0.1, stencil_width=1, procs=1)
        #center = CM(image, mode='concentric')
        center = (2463/2.,2527/2.)
        return center
        
        
    def as_shot(self):
        """
        Convert the CBF file to an ODIN shot representation.
        
        Returns
        -------
        cbf : odin.xray.Shot
            The CBF file as an ODIN shot.
        """
        
        # grid_list = (basis, shape, corner)
        # todo : do we need to center the image, changing corner?
        basis  = tuple( list(self.pixel_size) + [0.0] )
        shape  = tuple( list(self.intensities_shape) + [1] ) # add z dim
        corner = tuple( list(self.corner) + [0.0] )
        grid_list = [(basis, shape, corner )]
        
        b = xray.Beam(1e4, wavelength=self.wavelength)
        d = xray.Detector.from_basis(grid_list, self.path_length, b.k)
        s = xray.Shot(self.intensities.flatten().astype(np.float64), d)
        
        return s
        
        
class KittyH5(object):
    """
    A class that reads the output of kitty-enhanced psana.
    """
        
    def __init__(self, yaml_file, mode='asm'):
        """
        Load in a YAML file that describes a ShotSet collected on the LCLS.
        
        Parameters
        ----------
        yaml_file : str
            The path to the yaml file describing the shotset to load. Assumes
            the yaml file is of the following format:
            
              - data_file:
                photon_eV:
                ...
                
              - data_file:
                photon_eV:
                ...
              - ...
                
            where each list field corresponds to a different shot.
            
        mode : {'raw', 'asm'}
            Whether to load up the raw data or the assembled images.
        """
        
        logger.info('Kitty Loader -- locked onto shot handle: %s' % yaml_file)
        logger.info('Extracting %s-type data' % mode)
        
        f = open(yaml_file, 'r')
        self.yaml_data = yaml.load(f)
        f.close()
        
        self.descriptors = self.yaml_data['Shots']
        
        if mode in ['raw', 'asm']:
            self.mode = mode
        else:
            raise ValueError("`mode` must be one of {'raw', 'asm'}")
        
        return
    
    @property    
    def essential_fields(self):
        """
        A static property, a list of the essential data fields that must be
        provided for each shot to convert it into ODIN.
        """
        essentials = ['photon_eV', 'data_file', 'detector_mm']
        return essentials
    
        
    @property
    def num_descriptors(self):
        return len(self.descriptors)
    
        
    def convert(self, image_filter=None, max_shot_limit=None):
        """
        Perform the conversion from LCLS h5 files to an odin shotset.
        
        Optional Parameters
        -------------------
        image_filter : odin.xray.ImageFilter
            An image filter that will get applied to each shot.
        
        max_shot_limit : int
            If provided, will truncate the conversion after this many shots.
        
        Returns
        -------
        shotset : odin.xray.ShotSet
            A shotset containing all the shots described in the original yaml 
            file.
        """
        
        self.shot_list = []
        if image_filter:
            self.filter = image_filter
        else:
            self.filter = None
        
        if max_shot_limit:
            logger.info('Discarding all but the last %d shots' % max_shot_limit)
            n = max_shot_limit
        else:
            n = self.num_descriptors
            
        # convert the shots using the appropriate method
        for i in range(n):
            if self.mode == 'raw':
                s = self._convert_raw_shot(self.descriptors[i])
            elif self.mode == 'asm':
                s = self._convert_asm_shot(self.descriptors[i])
            self.shot_list.append(s)
    
        return xray.Shotset(self.shot_list)
        
    
    def _convert_raw_shot(self, descriptor):
        """
        Convert a Kitty-generated raw image h5 file to an odin.xray.Shot.
        """
        
        logger.info('Loading raw image in: %s' % descriptor['data_file'])
        
        for field in self.essential_fields:
            if field not in descriptor.keys():
                raise ValueError('Essential data field %s not in YAML file!' % field)
        
        energy = float(descriptor['photon_eV'])
        path_length = float(descriptor['detector_mm']) * 1000.0 # mm -> um
        
        # load all the relevant data from many h5s... (dumb)
        f = tables.File(descriptor['x_raw'])
        x = f.root.data.data.read()
        f.close()
        
        f = tables.File(descriptor['y_raw'])
        y = f.root.data.data.read()
        f.close()
        
        z = np.zeros_like(x)
        
        f = tables.File(descriptor['data_file'])
        path = descriptor['i_raw']
        i = f.getNode(path).read()
        f.close()
        
        # turn that data into a basis representation
        grid_list, flat_i = self._lcls_raw_to_basis(x, y, z, i)
        
        # generate the detector object               
        b = xray.Beam(100, energy=energy)
        d = xray.Detector.from_basis( grid_list, path_length, b.k )
        
        # generate the shot
        s = xray.Shot(flat_i, d)
        
        return s
        
                        
    def _convert_asm_shot(self, descriptor, x_pixel_size=109.9, 
                          y_pixel_size=109.9, bragg_peak_radius=458):
        """
        Convert a Kitty-generated assembled image h5 file to an odin.xray.Shot.
        """
        
        logger.info('Loading assembled image in: %s' % descriptor['data_file'])
        
        for field in self.essential_fields:
            if field not in descriptor.keys():
                raise ValueError('Essential data field %s not in YAML file!' % field)
                
        energy = float(descriptor['photon_eV']) / 1000.0        # eV -> keV
        path_length = float(descriptor['detector_mm']) * 1000.0 # mm -> um
        logger.debug('Energy:      %f keV' % energy)
        logger.debug('Path length: %f microns' % path_length)
        
        # extract intensity data
        f = tables.File(descriptor['data_file'])
        try:
            path = descriptor['i_asm']
        except KeyError as e:
            raise ValueError('Essential data field `i_asm` not in YAML file!')
        i = f.getNode(path).read()
        f.close()
        logger.debug('Read field %s in file: %s' % (descriptor['i_asm'],
                                                    descriptor['data_file']))
        
        # find the center (center is in pixel units)
        #dgeo = xray.DetectorGeometry(i)
        center = (853.,861.) #dgeo.center
        corner = ( -center[0] * x_pixel_size, -center[1] * y_pixel_size, 0.0 )
        logger.debug('Found center: %s' % str(center))
        
        # compile a grid_list object
        basis = (x_pixel_size, y_pixel_size, 0.0)
        shape = (i.shape[0], i.shape[1], 1)
        grid_list = [( basis, shape, corner )]
        
        # generate the detector object               
        b = xray.Beam(100, energy=energy)
        d = xray.Detector.from_basis( grid_list, path_length, b.k )

        logger.debug('Generated detector object...')
        
        # generate the shot
        s = xray.Shot(i.flatten(), d)
        
        return s
        
        
    def _lcls_raw_to_basis(self, x, y, z, intensities, x_asics=8, y_asics=8):
        """
        This is a specific function that converts a set of ASICS from the LCLS
        into a grid-basis vector representation. This representation is less
        flexible, but much more efficient computationally.
        
        Parameters
        ----------
        x,y,z : ndarray
            The x/y/z pixel coordinates. Assumed to be 2d arrays.

        x_asics, y_asics : int
            The number of asics in the x/y directions.

        intensities : ndarray, float
            The intensity data.

        Returns
        -------
        grid_list: list of tuples
            A basis vector representation of the detector pixels
                grid_list = [ ( basis, shape, corner ) ]
        """

        if not (x.shape == y.shape) and (x.shape == z.shape) and (x.shape == intensities.shape):
            raise ValueError('x, y, z, intensities shapes do not match!')

        # do some sanity checking on the shape
        s = x.shape
        if not s[0] % x_asics == 0:
            raise ValueError('`x_asics` does not evenly divide the x-dimension!')
        if not s[1] % y_asics == 0:
            raise ValueError('`y_asics` does not evenly divide the y-dimension!')

        # determine the spacing
        x_spacing = s[0] / x_asics
        y_spacing = s[1] / y_asics

        # grid list to hold the output grid_list = [ ( basis, shape, corner ) ]
        grid_list = []
        flat_intensities = np.zeros( np.product(intensities.shape) )

        # iterate through each asic and extract the basis vectors
        for ix in range(x_asics):
            for iy in range(y_asics):

                # initialize data structures
                basis = []

                # the indicies to slice the array on
                x_start  = x_spacing * ix
                x_finish = x_spacing * (ix+1)

                y_start  = y_spacing * iy
                y_finish = y_spacing * (iy+1)

                # slice
                xa = x[x_start:x_finish,y_start:y_finish]
                ya = y[x_start:x_finish,y_start:y_finish]
                za = z[x_start:x_finish,y_start:y_finish]
                ia = intensities[x_start:x_finish,y_start:y_finish]

                # determine the pixel spacing
                for a in [xa, ya]:
                    s1 = np.abs(a[0,0] - a[0,1])
                    s2 = np.abs(a[0,0] - a[1,0])
                    basis.append( max(s1, s2) )

                # add the z-basis
                basis.append(0.0)
                assert len(basis) == 3
                basis = tuple(basis)

                # assemble the grid_tuple
                shape = (x_spacing, y_spacing, 1)

                corner = ( xa.flatten()[np.argmin(xa)], 
                           ya.flatten()[np.argmin(ya)], 
                           za[0,0] )

                grid_tuple = (basis, shape, corner)
                grid_list.append(grid_tuple)

                # store the flatten-ed intensities in the correct order
                spacing = x_spacing * y_spacing
                start  = spacing * (ix + iy*x_asics)
                finish = spacing * (ix + iy*x_asics + 1)
                flat_intensities[start:finish] = ia.flatten()

        assert len(grid_list) == x_asics * y_asics

        return grid_list, flat_intensities
        
                
        
class CXI(object):
    """
    A parser for the coherent x-ray imaging file format.
    
    This format is the standard for LCLS experiments; details can be found here:
    
        http://cxidb.org/
    
    This class parses the file and serves an odin.xray.Shotset, along with some
    additional metadata associated with the run.
    
    Initilization Parameters
    ------------------------
    filename : str
        The HDF5 file path.
    """
    
    def __init__(self, filename):
        
        # EEEK
        raise NotImplementedError('CXI parser not complete, sorry')
        
        self.filename = filename
        self._get_root()
        
        self.shots = []
        for entry_group in self._get_groups('entry'):
            self.shots.append( self._generate_shot_from_entry(entry_group) )
            
        self.shotset = xray.Shotset( self.shots )
        
        
    def _get_root(self):
        h5 = tables.openFile(self.filename, mode='r')
        self.h5 = h5
        self.root = self.h5.root
        
        
    def _get_groups(self, name, root='/'):
        """
        Locates groups in the HDF5 file structure, beneath `root`, with name
        matching `name`.
        
        Returns
        -------
        groups : list
            A list of pytables group objects
        """
        groups = []
        for g in self.h5.walkGroups(root):                
            gname = g.__str__().split('/')[-1].split()[0]
            if gname.find(name) == 0:
                groups.append(g)
        return groups
            
            
    def _get_nodes(self, name, root='/', strict=False):
        """
        Locates nodes in the HDF5 file structure, beneath `root`, with name
        matching `name`.

        Returns
        -------
        nodes : list
            A list of pytables nodes objects
        """
        nodes = []
        for n in self.h5.walkNodes(root):                
            nname = n.__str__().split('/')[-1].split()[0]
            if not isinstance(n, tables.link.SoftLink):
                if strict:
                    if nname == name:
                        nodes.append(n)
                else:
                    if nname.find(name) == 0:
                        nodes.append(n)
        return nodes
        
                    
    def _extract_energy(self, entry):
        """
        Scrape the following important metadata out of the .cxi file:
        -- beam energy
        -- 
        """
        energy_nodes = self._get_nodes('energy', root=entry, strict=True)
        if len(energy_nodes) > 1:
            raise RuntimeError('Ambiguity in beam energy... %d energy nodes found' % len(energy_nodes))
        energy = float(energy_nodes[0].read()) / 1000.0
        beam = xray.Beam(1., energy=energy) # todo: compute flux
        return beam
        
	    
    def _detector_group_to_array(self, detector_group):
        """
        Generate an array of the xyz coordinates of the pixels using the rules 
        implicit in the CXI format.

        Parameters
        ----------
        size : tuple
            (number_x_pixels, number_y_pixels)
        x_basis, y_basis : float
            The size of the pixels in the x/y direction
    
        Returns
        -------
        xyz : ndarray, float
            An n x 3 array of the pixel positions.
        intensities : ndarray, float
            An n-len array of the intensities at each pixel
        metadata : dict
            A dictionary of any extra metadata extracted from the detector 
            group object
        """

        metadata = {}
        # ----- todo TJL not sure if these are in all CXI files? ---------
        #       put them --> metadata
        #is_fft_shifted = bool(detector_group.is_fft_shifted.read())
        #image_center = detector_group.image_center.read()
        #path_length = float(detector_group.distance.read())
        #mask = detector_group.mask.read()
        # ----------------------------------------------------------------

        # sometimes the intensities are not underneath the detector instance,
        # so we have to check where they are...
        if hasattr(detector_group, 'data'):
            intensities = np.real( detector_group.data.read() )
        else:
            data_nodes = self._get_nodes('data', strict=True)
            if len(data_nodes) > 1:
                # if we're here, we can't uniquely associate a 'data' entry with
                #  the detector pixel positions...
                raise RuntimeError('Ambiguous location of intensities in CXI '
                                    'file... cannot proceed.')
            else:
                intensities = np.real( data_nodes[0].read() )
        
        # the intensities come out in a 2-D array -- we'll want them flatttened
        size = intensities.shape
        intensities = intensities.flatten()

        x_pixel_size = float(detector_group.x_pixel_size.read())
        y_pixel_size = float(detector_group.y_pixel_size.read())

        # try to find basis vectors for pixel positions
        # if none exist, convention is to use the pixel sizes
        # TJL todo CHECK THE NAMES BELOW!!!
        if hasattr(detector_group, 'x_basis'):
            x_basis = float(detector_group.x_basis.read())
        else:
            x_basis = x_pixel_size
        if hasattr(detector_group, 'y_basis'):
            y_basis = float(detector_group.y_basis.read())
        else:
            y_basis = y_pixel_size

        x_pixels = np.arange(0, x_basis*size[0], x_basis)
        y_pixels = np.arange(0, y_basis*size[1], y_basis)
        assert len(x_pixels) == size[0]
        assert len(y_pixels) == size[1]

        z = np.zeros( len(x_pixels) * len(y_pixels) )
        x, y = np.meshgrid(x_pixels, y_pixels)
        x = x.flatten()
        y = y.flatten()

        xyz = np.vstack((x, y, z)).transpose()

        # if there is a 'corner_position' attr, translate
        if hasattr(detector_group, 'corner_position'):
            xyz += detector_group.corner_position.read()

        return xyz, intensities, metadata
        
        
    def _generate_shot_from_entry(self, entry):

        # loop over all the detector elements and piece them together
        xyz = []
        intensities = []
        metadatas = []
        for detector_group in self._get_groups('detector', root=entry):
            x, i, m = self._detector_group_to_array(detector_group)
            xyz.append(x)
            intensities.append(i)
            metadatas.append(m)
        
        xyz = np.concatenate(xyz)
        intensities = np.concatenate(intensities)

        # instantitate a detector
        # todo NEED: logic for path_length
        path_length = 1.0
        beam = self._extract_energy(entry)
        d = xray.Detector(xyz, path_length, beam.k)

        # generate a shot instance and add it to our list
        return xray.Shot(intensities, d)



