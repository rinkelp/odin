
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
        
        
    def test_basic(self):
        # assert that the one circle in test1 gets picked up
        maxima = self.CM(self.image, concentric=False)
        print "maxima", maxima
        assert len(maxima) == 1
        assert maxima[0] == (85.0, 155, 143) # confirmed visually
     
        
    def test_many_circles(self):
        
        image = plt.imread(ref_file('chough-test2.png'))
        
        maxima = self.CM(image, concentric=False)
                                  
        print "many circles:", maxima
        assert False
        
        
    def test_concentric(self):
        maxima = self.CM(self.image, concentric=True)
        print "maxima", maxima
        assert maxima == (155, 143)