
"""
tests for odin/src/python/math.py
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose

from odin import math as om
from odin.testing import ref_file, skip, expected_failure


class TestHough(object):
    
    def setup(self):
        self.image = plt.imread(ref_file('chough-test.png'))
        self.CM = om.CircularHough(radii=np.arange(70,95,3))
             
    @expected_failure
    def test_all(self):
        maxima = self.CM(self.image, mode='all')                          
        assert len(maxima) == 1
        assert_allclose(maxima[0], (85.0, 155, 143))
        
    def test_sharpest(self):
        maxima = self.CM(self.image, mode='sharpest')                          
        assert_allclose(maxima, (85.0, 155, 143))
        
    def test_concentric(self):
        maxima = self.CM(self.image, mode='concentric')
        assert_allclose(maxima, (152, 150)) # is just a little off
        
    @skip
    def test_all_on_many_img(self):
        image = plt.imread(ref_file('chough-test2.png'))
        maxima = self.CM(self.image, mode='all')                          
        print "many circles:", maxima
        assert False