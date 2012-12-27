
"""
tests for odin/src/python/math.py
"""

import scipy
import matplotlib.pyplot as plt

from odin import math as om
from odin.testing import ref_file, skip


class TestHough(object):
    
    def setup(self):
        self.image = plt.imread(ref_file('chough-test.png'))
        self.CM = om.CircularHough()
        
    @skip
    def test_basic(self):
        # assert that the one circle in test1 gets picked up
        maxima = self.CM(self.image, radii=20, threshold=1e-4, stencil_width=1,
                         concentric=False)
        assert len(maxima) == 1
        assert maxima[0] == (85.0, 155, 143) # confirmed visually
     
    @skip 
    def test_many_circles(self):
        
        image = plt.imread(ref_file('chough-test2.png'))
        
        maxima = self.CM(image, radii=20, threshold=1e-4, 
                         stencil_width=1, concentric=False)
                                  
        print "many circles:", maxima
        
    @skip    
    def test_concentric(self):
        maxima = self.CM(self.image, radii=20, threshold=1e-4, 
                         stencil_width=1, concentric=True)    
        assert maxima == (155, 143)