
from odin import xray
from odin.bcinterp import Bcinterp
from odin.testing import skip, ref_file

import numpy as np
from scipy import interpolate
from numpy.testing import assert_allclose, assert_almost_equal
 

class TestBcinterp():
    
    def setup(self):
                
        # initialize parameters
        dim1 = 100
        dim2 = 100
        
        self.vals = np.random.randn(dim1 * dim2)
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
                                         
    def test_array_evaluation(self):
        ip = self.interp.evaluate(self.new_x, self.new_y)
        assert( np.sum( np.abs( ip - self.ref ) ) / float(len(ip)) < 0.5 )
        
if __name__ == '__main__':
    test = TestBcinterp()
    test.setup()
    test.test_array_evaluation()
