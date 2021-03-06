import numpy as np
cimport numpy as np

  
cdef extern from "bcinterp.hh":
    cdef cppclass C_Bcinterp "Bcinterp":
        C_Bcinterp(int Nvals, float *vals, float x_space_, float y_space_,
            int Xdim_, int Ydim_, float x_corner_, float y_corner_) except +
        float evaluate_point(float x, float y)
        void evaluate_array(int dim_xa, float *xa, int dim_ya, float *ya, 
                            int dim_za, float *za)
        float x_space
        float y_space
        float x_corner
        float y_corner
        int Xdim
        int Ydim
        
      
cdef class Bcinterp:
        
    cdef C_Bcinterp * c
    
    def __init__(self, vals, x_space, y_space, Xdim, Ydim, x_corner, y_corner):
        """
        Generate a bicubic interpolator, given a set of observations `values`
        made on a 2D grid.
    
        Parameters
        ----------
        values : ndarray, float
            The observed values. Can either be a one- or two-dimensional array.
            Note that x is assumed to be the fast scan direction.
    
        x_spacing, y_spacing : float
            The grid spacing, in the x/y direction
    
        x_dim, y_dim : int
            The size of the grid in the x/y dimension
            
        x_corner, y_corner : float
            The location of the bottom left corner of the image.
    
        Returns
        -------
        odin.math.Bcinterp
    
        See Also
        --------
        Bcinterp.evaluate : function
            Evaluate the interpolated function at one or more points
        """
        
        if len(vals.shape) == 1:
            pass
        elif len(vals.shape) == 2:
            vals = vals.flatten()
        else:
            raise ValueError('`vals` must be a one- or two-dimensional array')
        
        cdef np.ndarray[ndim=1, dtype=np.float32_t] v
        v = np.ascontiguousarray(vals, dtype=np.float32)
        
        if not (type(x_space) == float) and (type(y_space) == float):
            raise TypeError('`x_space`, `y_space` must be type: float')
        if not (type(Xdim) == int) and (type(Ydim) == int):
            raise TypeError('`Xdim`, `Ydim` must be type: int')
        if not (type(x_corner) == float) and (type(y_corner) == float):
            raise TypeError('`x_corner`, `y_corner` must be type: float')
         
        if not np.product(vals.shape) == Xdim * Ydim:
            raise ValueError('`vals` must have `Xdim` * `Ydim` total pixels')
        
        self.c = new C_Bcinterp(len(vals), &v[0], x_space, y_space,
                                Xdim, Ydim, x_corner, y_corner)
                
    def __dealloc__(self):
        del self.c
        
    def _evaluate_array(self, np.ndarray[ndim=1, dtype=np.float32_t] x,
            np.ndarray[ndim=1, dtype=np.float32_t] y):
        assert len(x) == len(y)
        
        cdef np.ndarray[ndim=1, dtype=np.float32_t] z = np.zeros_like(x)
        self.c.evaluate_array(len(x), &x[0], len(y), &y[0], len(z), &z[0])
        return z
        
    def _evaluate_point(self, float x, float y):
        pt = self.c.evaluate_point(x, y)
        return pt
        
    def evaluate(self, x, y):
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
        
        x_min = self.c.x_corner
        x_max = self.c.x_corner + (self.c.Xdim-1) * self.c.x_space
        y_min = self.c.y_corner
        y_max = self.c.y_corner + (self.c.Ydim-1) * self.c.y_space
        
        if np.isscalar(x) and np.isscalar(y):
            if (x < x_min) or (x > x_max):
                raise ValueError('x_point out of range of convex hull of '
                                 'interpolation')
            elif (y < y_min) or (y > y_max):
                raise ValueError('y_point out of range of convex hull of '
                                 'interpolation')
            else:
                z = self._evaluate_point(x, y)
            
        elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            if np.any((x < x_min)) or np.any((x > x_max)):
                raise ValueError('x_point out of range of convex hull of '
                                 'interpolation')
            elif np.any((y < y_min)) or np.any((y > y_max)):
                raise ValueError('y_point out of range of convex hull of '
                                 'interpolation')
            else:
                x = np.ascontiguousarray(x, dtype=np.float32)
                y = np.ascontiguousarray(y, dtype=np.float32)
                z = self._evaluate_array(x, y)
                z = z.astype(np.float64)
            
        else:
            raise TypeError('x,y must be floats for arrays of floats')
            
        return z
