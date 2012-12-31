import numpy as np
cimport numpy as np

  
cdef extern from "bcinterp.hh":
    cdef cppclass C_Bcinterp "Bcinterp":
        C_Bcinterp(int Nvals, double *vals, double x_space_, double y_space_,
            int Xdim_, int Ydim_, double x_corner_, double y_corner_) except +
        double evaluate_point(double x, double y)
        void evaluate_array(int dim_xa, double *xa, int dim_ya, double *ya, 
                            int dim_za, double *za)
        double x_space
        double y_space
        double x_corner
        double y_corner
        int Xdim
        int Ydim
        
      
cdef class Bcinterp:
        
    cdef C_Bcinterp* c
    
    def __cinit__(self, np.ndarray[ndim=1, dtype=np.double_t] vals, double x_space,
            double y_space, int Xdim, int Ydim, double x_corner, double y_corner):
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
        self.c = new C_Bcinterp(len(vals), &vals[0], x_space, y_space,
                                Xdim, Ydim, x_corner, y_corner)
                
    def __dealloc__(self):
        del self.c
        
    def _evaluate_array(self, np.ndarray[ndim=1, dtype=np.double_t] x,
            np.ndarray[ndim=1, dtype=np.double_t] y):
        assert len(x) == len(y)
        cdef np.ndarray[ndim=1, dtype=np.double_t] z = np.zeros_like(x)
        self.c.evaluate_array(len(x), &x[0], len(y), &y[0], len(z), &z[0])
        return z
        
    def _evaluate_point(self, double x, double y):
        return self.c.evaluate_point(x, y)
        
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
                z = self._evaluate_array(x, y)
            
        else:
            raise TypeError('x,y must be floats for arrays of floats')
            
        return z
