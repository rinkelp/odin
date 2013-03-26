
"""
tests for odin/src/python/math2.py
"""

import scipy
import numpy as np
from scipy.ndimage import imread
from numpy.testing import assert_allclose

from odin import parse
from odin import math2 as om
from odin.testing import ref_file, skip, expected_failure

from scipy.ndimage import imread

class TestHough(object):
    
    def setup(self):
        self.image = np.array(imread(ref_file('chough-test.png')))
        self.CM = om.CircularHough(radii=np.arange(75,87,1), procs='all')
    
    @skip         
    def test_all(self):
        # todo : figure out why this test only partially passes:
        # many false positives
        maxima = self.CM(self.image, mode='all')
        print "all:", maxmia
        assert (85.0, 155, 143) in maxima
        
    @skip
    def test_sharpest(self):
        maxima = self.CM(self.image, mode='sharpest')                          
        assert_allclose(maxima, (85.0, 156, 145))
    
    @skip
    # for some reason this seems to be stochastic?!?
    # todo : fix that. its not good.
    def test_concentric(self):
        maxima = self.CM(self.image, mode='concentric')
        assert_allclose(maxima, (155, 143))
        
    @skip
    def test_all_on_many_img(self):
        image = imread(ref_file('chough-test2.png'))
        CM = om.CircularHough(radii=np.arange(10,40,2))
        maxima = CM(image, mode='all')                          
        print "many circles:", maxima
        
        # the reference was confirmed visually TJL 12.27.12, the last
        # circle is, in fact, a false positive... (not sure how to fix it)
        ref = [(20, 62, 272), (20, 63, 168), (20, 64, 64), (22, 59, 376), 
               (22, 60, 482), (26, 57, 691), (26, 58, 587), (26, 59, 589), 
               (30, 56, 795), (34, 52, 952)]
               
        assert_allclose(maxima, ref)

    @skip
    def test_xray_rings(self):
        """ test the Hough transform on some real data """
        # final result confirmed visually
        # todo : test against derek
        cbf = parse.CBF("reference/test1.cbf")
        image = cbf.intensities.reshape( cbf.intensities_shape )
        CM = om.CircularHough(radii=np.arange(70,95,3))
        center = CM(self.image, mode='concentric')
        
    @skip
    def test_parallel(self):
        """ test: ensure parallel & serial Hough are consistent """
        parallel_maxima = self.CM(self.image, mode='sharpest')
        sCM = om.CircularHough(radii=np.arange(70,95,3), procs=1)
        serial_maxima = self.CM(self.image, mode='sharpest')
        assert_allclose(parallel_maxima, serial_maxima)
        
        
        
        
