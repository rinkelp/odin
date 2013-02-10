
from odin import xray
from odin.interp import Bcinterp
from odin.testing import skip, ref_file

import numpy as np
from scipy import interpolate
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_almost_equal
 

class TestBcinterp():
    
    def setup(self):
                
        # initialize parameters
        dim1 = 100
        dim2 = 100
        
        self.vals = np.abs(np.random.randn(dim1 * dim2))
        self.x_space = 0.1
        self.y_space = 0.1
        self.Xdim = dim1
        self.Ydim = dim2
        self.x_corner = 0.0
        self.y_corner = 0.0
                
        self.interp = Bcinterp( self.vals, 
                                self.x_space, self.y_space, 
                                self.Xdim, self.Ydim,
                                self.x_corner, self.y_corner )
                                
        # generate a reference
        x = np.arange(self.x_corner, self.x_corner + self.Xdim*self.x_space, self.x_space)
        y = np.arange(self.y_corner, self.y_corner + self.Ydim*self.y_space, self.y_space)
        
        xx, yy = np.meshgrid(x,y)
        
        self.new_x = np.arange(0.0, 8.0, .01) + 0.1
        self.new_y = np.arange(0.0, 8.0, .01) + 0.1
        self.ref = interpolate.griddata( np.array([xx.flatten(),yy.flatten()]).T, 
                                         self.vals, 
                                         np.array([self.new_x, self.new_y]).T,
                                         method='cubic' )
        
    def test_for_smoke(self):
        ip = self.interp.evaluate(self.new_x, self.new_y)
        if np.all( ip == 0.0 ):
            print "Interpolator not working, likely cause: OMP failure."
            print "Try reinstalling with no OMP: python setup.py install --no-openmp"
            raise Exception()
    
    def test_point_evaluation(self):
        x = float(self.new_x[1])
        y = float(self.new_y[1])
        i = self.interp._evaluate_point(x,y)
        assert_allclose( i, self.ref[1], rtol=0.5 )

    def test_point_vs_known(self):
        interp = Bcinterp( np.arange(1000**2),
                           0.1, 0.1, 1000, 1000, 0.0, 0.0 )
        i = interp.evaluate(1.01, 1.01)
        assert_almost_equal(i, 10110.1, decimal=0)
                                       
    def test_array_evaluation(self):
        ip = self.interp.evaluate(self.new_x, self.new_y)
        assert np.sum( np.abs(ip - self.ref) < 0.3 ) > 600
        
if __name__ == '__main__':
    test = TestBcinterp()
    test.setup()
    test.test_array_evaluation()
