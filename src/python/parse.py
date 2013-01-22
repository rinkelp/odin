
"""
Various parsers:
-- CBF (crystallographic binary format)
-- CXI (coherent xray imaging format)
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

import inspect
import tables
import re
import hashlib
import yaml
import tables
from base64 import b64encode

import numpy as np

from odin import xray
from odin.math import CircularHough

ENABLE_CBF = True
try:
    import pycbf
except ImportError as e:
    ENABLE_CBF = False
    logger.warning('Could not find dependency `pycbf`. Install libcbf and'
                   ' pycbf. See the ODIN documentation for more details.')


class CBF(object):
    """
    A class for parsing CBF files. Depends on pycbf, which is a python-SWIG
    wrapped cbflib.
    """
    
    # --------------------------------------------------------------------------
    # Warning: Rant Below
    #
    # cbflib is a peice of junk. The file format itself, while probably
    # unnecessary, is fine. The library, however, is unnecessarily convoluted,
    # obtuse, and painful to use. This should be a straightforward thing, i.e.
    # load file and be done, but as evidenced in the code below, how to use the
    # library involves voodoo that is COMPLETELY UNDOCUMENTED, and requires you
    # to read the fucking C source code to find out what's going on. Whoever
    # designed this peice of shit is clearly an engineer who needs to learn what
    # humans are, and how to be a decent person.
    #
    # Fortunately, someone SWIG wrapped this mess, which at least makes the
    # guess-and-check process necessary to use the code faster. Unfortunately
    # the python version, pycbf, is also completely document/comment free, and
    # of course it reflects the user-misleading pathologies of the parent 
    # library.
    #
    # I can only pray that the code below works. If you run into problems send
    # TJ an email at <tjlane@stanford.edu> and we can work together to cut
    # through the BS.
    #
    # Stuff shouldn't be this hard....
    #
    # OK, hope you skipped that. Productive code below.
    # --------------------------------------------------------------------------
    
    def __init__(self, filename):
        logger.info('Reading: %s' % filename)
        if not ENABLE_CBF:
            raise ImportError('Require `libcbf` and `pycbf` to parse CBFs')
        self.filename = filename
        
        self._get_header_data()
        self.intensity_dtype = self._convert_dtype(self._info['X-Binary-Element-Type'])
        self._get_array_data()
        
        # currently disabled, see comment in functoin
        # if self.md5:
        #     self._check_md5()
            
        logger.debug('loaded file')
        
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
        d, unit = self._info['Detector_distance'].split() # assume units are the same for all dims
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
        
    def _convert_dtype(self, dtype_str):
        """
        Converts `dtype_str`, straight from the cbf file, to the right numpy
        dtype
        """
        
        # TJL: I'm just guessing the names for most of these....
        # the cbflib docs are useless!!
        conversions = {'"signed 32-bit integer"' : np.int32,
                       '"unsigned 32-bit integer"' : np.uint32,
                       '"32-bit float"' : np.float32,
                       '"64-bit float"' : np.float64}
        
        try:
            dtype = conversions[dtype_str]
        except KeyError as e:
            raise ValueError('Binary-Element-Type: %s has no know numpy '
                             'counterpart. Contact the dev team if you believe '
                             'this is wrong -- it may be an unexpected string.'
                             % dtype_str)
        
        return dtype
        
        
    def _get_header_data(self):
        """
        Parse the flat-text header section for all the useful self._info
        """
        
        logger.debug('Reading header info...')
        
        self._info = {} # everything gets dumped here
        f = open(self.filename, 'rb')
        st = str()
        lines = list()
        
        # read char-by-char until we get to the binary header
        # the first flat text section has to do with the detector & run params
        loop_counter = 0 # saftey-check to prevent infinite loops
        loop_cutoff = int(1e8)
        
        while True:
            st += f.read(1)
            if st == "--CIF-BINARY-FORMAT-SECTION--\r\n":
                st = ""
                break
            elif st.endswith("\n"):
                lines.append(st)
                split = st.strip().split()
                if len(split) > 0:
                    if split[0] == "#":
                        k = split[1].strip().lstrip(':')
                        self._info[k] = ' '.join(split[2:])
                st = ""
            loop_counter += 1
            if loop_counter > loop_cutoff:
                raise RuntimeError('No break from parse loop -- file corrupted!')

        # there is a second flat-text section that describes the binary data
        logger.debug("Reading binary header")
        loop_counter = 0
        st = ""
        
        while True:
            st += f.read(1)
            if (st == "\r\n"): # or (st == "\r") or (st == "\n"):
                break
            elif st.endswith("\n"): # or st.endswith("\r"):
                lines.append(st)
                split = st.strip().split(':')
                if len(split) == 2:
                    try:
                        self._info[split[0].strip()] = int(split[1])
                    except:
                        self._info[split[0].strip()] = split[1].strip()
                st = ""
            if loop_counter > loop_cutoff:
                raise RuntimeError('No break from parse loop -- file corrupted!')

        f.close()
        return

        
    def _get_array_data(self):
        """
        Finds the numerical data in the CBF file and injects it into intensities.
        Uses libcbf & pycbf to parse this info.
        """
        
        logger.debug('Reading binary intensities...')
        
        # `_handle` is the c-struct object that holds all the CBF-file info
        self._handle = pycbf.cbf_handle_struct()
        self._handle.read_file(self.filename, pycbf.MSG_DIGEST)
        self._handle.rewind_datablock()
        
        # some cbflib voodoo...
        self._n_blocks = self._handle.count_datablocks()
        self._handle.select_datablock(0)
        self._block_name = self._handle.datablock_name()
        logger.debug('Selected block %d, %s' % (0, self._block_name))
        
        self._handle.rewind_category()
        self.n_categories = self._handle.count_categories()
        
        # I believe each "category" may contain a different image
        # todo dblchk
        for i in range(self.n_categories):
            
            self._handle.select_category(i)
            category_name = self._handle.category_name()
            logger.debug("Reading category %d, %s" % (i, category_name))

            n_rows = self._handle.count_rows()
            if n_rows > 1:
                logger.warning("More than one `row` in CBF, parsing first only...")
            elif n_rows == 0:
                raise RuntimeError('No rows in CBF! Cannot parse.')
                
            n_cols = self._handle.count_columns()
            logger.debug("rows/cols: %d/%d" % (n_rows, n_cols))

            # step through each "column", I don't even know what those are...
            self._handle.rewind_column()
            while True:
                logger.debug("column: %s" % self._handle.column_name())
                try:
                   self._handle.next_column()
                except:
                   break
                
            for k in range(n_cols):
                col_name = self._handle.column_name()
                self._handle.select_column(k)
                
                typeofvalue = self._handle.get_typeofvalue()
                logger.debug("col: %d, name: %s, type: %s" % (k, col_name, typeofvalue))

                if typeofvalue.find("bnry") > -1:
                    logger.debug("Found binary data")
                    try:
                        s = self._handle.get_integerarray_as_string()
                    except AttributeError as e:
                        logger.critical("ERROR: %s" % e)
                        raise AttributeError('CBF PARSE ERROR: Could not find intensity data in file!')
                    d = np.fromstring(s, dtype=self.intensity_dtype)
                    self.intensities = d.reshape(*self.intensities_shape) # (slow, fast)
                else:
                    pass
                    #value = self._handle.get_value()
                    #logger.debug("column value: %s", str(value))

                        
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
        Find the center of any Bragg rings (aka the location of the x-ray beam)
        using a Hough Transform.
        
        Returns
        -------
        center : tuple of ints
            The indicies of the pixel nearest the center of the Bragg peaks.
        """
        CM = CircularHough(radii=10, threshold=0.1, stencil_width=1, procs=1)
        center = CM(image, mode='concentric')
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
        
        b = xray.Beam(wavelength=self.wavelength, flux=100)
        d = xray.Detector.from_basis(grid_list, self.path_length, b.k)
        s = xray.Shot(self.intensities.flatten().astype(np.float64), d)
        
        return s
        
        
class KittyHD5(object):
    """
    A class that reads the output of kitty-enhanced psana.
    """
        
    def __init__(self, yaml_file):
        """
        Load in a YAML file that describes a ShotSet collected on the LCLS.
        
        Parameters
        ----------
        yaml_file : str
            The path to the yaml file describing the shotset to load. Assumes
            the yaml file is of the following format:
            
              - filename:
                energy:
                ...
                
              - filename:
                energy:
                ...
                
                ...
                
            where each list field corresponds to a different shot.
        """
                
        f = open(yaml_file, 'r')
        self.yaml_data = yaml.load(f)
        
        # see if there is a single pixel map to be used for the entire shotset
        # and if so, store it as a `master_detector` attribute. If not, we'll
        # generate a different detector for each shot
        try:
            self.master_detector = self._convert_detector(self.yaml_data['geom_file'],
                                                          self.yaml_data['photon_eV'],
                                                          self.yaml_data['detector_mm'])
            self.is_master_detector = True
        except KeyError as e:
            logger.info('YAML file lacks field: `geom_file`')
            logger.info('Looking for pixel map data associated with each shot entry...')
            self.is_master_detector = False
    
    
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
            for i in range(max_shot_limit):
                self.shot_list.append(self._convert_shot(self.descriptors[i]))
        else:
            for i in range(self.num_descriptors):
                s = self._convert_shot(self.descriptors[i])
                self.shot_list.append(s)
    
        return xray.Shotset(self.shot_list)
        
    
    def _convert_detector(self, detector_h5_file, energy, path_length):
        """
        Convert a pixel map, written as an LCLS h5 file, to an odin Detector
        object. Stores that detector in 
        """
        
        f = tables.File(detector_h5_file)
        
        xyz = vstack([ f.root.x.read().flatten(), 
                       f.root.y.read().flatten(), 
                       f.root.z.read().flatten() ]).T
                       
        b = xray.Beam(energy=energy)
        d = xray.Detector( xyz, path_length, b.k )
        
        return d
        
                        
    def _convert_shot(self, descriptor):
        """
        Convert a single shot from LCLS H5 into an odin.shot
        
        Returns
        -------
        shot : odin.xray.Shot
            The converted shot corresponding to `descriptor`
        """
        
        for field in self.essential_fields:
            if field not in descriptor.keys():
                raise ValueError('Essential data field %s not in YAML file!' % field)
                
        # load up the intensity data from the file indicated in the descriptor
        f = tables.File(descriptor['data_file'])
        intensities = f.root.data.data.read().flatten()
        
        # if we're using an image filter, apply it
        if self.filter:
            intensities = self.filter(intensities)
        
        # figure out what detector to use
        if self.is_master_detector:
            d = self.master_detector
        else:            
            if 'geom_file' not in descriptor.keys():
                raise ValueError('Essential data field `geom_file` not in YAML file!')
            d = self._convert_detector(descriptor['geom_file'],
                                       descriptor['photon_eV'],
                                       descriptor['detector_mm'])
        
        # generate a detector object 
        shot = xray.Shot(intensities, d)
        
        return shot
        
        
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



