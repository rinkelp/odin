
"""
Tests for /src/python/parse.py
"""

import os, sys
import warnings
from nose import SkipTest

from odin import parse, xray
from odin.testing import skip, ref_file, gputest, expected_failure
from mdtraj import trajectory, io


class TestCBF(object):
    
    def setup(self):
        self.cbf = parse.CBF(ref_file('test1.cbf'))
        
    #def test_smoke_on_load_files(self):
        #out = parse.CBF(ref_file('test2.cbf'))
        # todo : could use more cbf example files (from a modern v. of cbflib)
        
    def test_intensities_shape(self):
        s = self.cbf.intensities_shape
        assert s == (2527, 2463)
        
    def test_pixel_size(self):
        x = self.cbf.pixel_size
        assert x == (0.000172, 0.000172)
        
    def test_path_length(self):
        l = self.cbf.path_length
        assert l == 0.2
        
    def test_wavelength(self):
        l = self.cbf.wavelength
        assert l == 0.7293
        
    def test_polarization(self):
        p = self.cbf.polarization
        assert p == 0.99
    
    @expected_failure    
    def test_center(self):
        c = self.cbf.center
        assert False
    
    @expected_failure    
    def test_corner(self):
        c = self.cbf.corner
        x = self.cbf.pixel_size
        y = self.cbf.center
        ref = (x[0] * float(y[0]), x[1] * float(y[1]))
        assert ref == c
        assert False

    def test_as_shot(self):
        s = self.cbf.as_shot()
        assert isinstance(s, xray.Shot)
        
        
class TestCXI(object):
    def setup(self):
        pass