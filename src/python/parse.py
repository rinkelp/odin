
"""
Various parsers.
"""


import inspect
import tables
import re
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from odin import xray


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



