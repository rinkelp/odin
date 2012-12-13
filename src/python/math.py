
"""
math.py

Various mathematical functions and operations.
"""

import numpy as np

from odin import bcinterp as cbc


class Bcinterp(object):
    """ Provides functionality for bicubic interpolation. A thin wrapper around
        SWIG-wrapped c++ """
        
    def __init__(self, values, x_spacing, y_spacing, x_dim, y_dim):
        """
        Generate a bicubic interpolator, given a set of observations `values`
        made on a 2D grid.
        
        Parameters
        ----------
        values : ndarray, float
            The observed value 
        
        x_spacing, y_spacing : float
            The grid spacing, in the x/y direction
        
        x_dim, y_dim : int
            The size of the grid in the x/y dimension
        
        Returns
        -------
        odin.math.Bcinterp
        
        See Also
        --------
        Bcinterp.ev : function
            Evaluate the interpolated function at one or more points
        """
        
        if not (type(x_spacing) == float) and (type(y_spacing) == float):
            raise TypeError('x_spacing/y_spacing must be type float')
            
        if not (type(x_dim) == int) and (type(y_dim) == int):
            raise TypeError('x_dim/y_dim must be type int')
            
        if not values.dtype == np.float:
            values = values.astype(np.float)
            
        values = values.flatten()
        if not len(values) == x_dim * y_dim:
            raise ValueError('`values` must be of size x_dim * y_dim')
        
        self.interpolator = cbc.Bcinterp(values, x_spacing, y_spacing,
                                         x_dim, y_dim)
                                         
        return
        
    def ev(self, x, y):
        """
        Evaluate the interpolated grid at point(s) (x,y)
        
        Parameters
        ----------
        x,y : float or ndarray, float
            The points at which to evaluate the interpolation
        
        Returns
        -------
        z : float or ndarray, float
            The values of the interpolation
        """
        
        if (type(x) == float) and (type(y) == float):
            z = self.interpolator.evaluate_point(x, y)
        
        elif (type(x) == np.ndarray) and (type(y) == np.ndarray):
            if not x.dtype == np.float:
                x = x.astype(np.float)
            if not y.dtype == np.float:
                y = y.astype(np.float)
                
            x = x.flatten()
            y = y.flatten()
                
            if not len(x) == len(y):
                raise ValueError('x, y must be same length')
                
            z = self.interpolator.evaluate_array(x, y, len(x))
        
        else:
            raise TypeError('x/y must be floats or ndarray of type np.float')
            
        return z


def fft_acf(data):
    '''
    Return the autocorrelation of a 1D array using the FFT
    Note: the result is normalized
    
    Parameters
    ----------
    data : ndarray, float, 1D
        Data to autocorrelate
        
    Returns
    -------
    acf : ndarray, float, 1D
        The autocorrelation function
    '''
    data = data - np.mean(data)
    result = signal.fftconvolve(data, data[::-1])
    result = result[result.size / 2:] 
    acf = result / result[0]
    return acf
    
    
