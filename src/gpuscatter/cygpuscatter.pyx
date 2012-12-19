import numpy as np
cimport numpy as np

  
cdef extern from "gpuscatter_mgr.hh":
    cdef cppclass C_Bcinterp "GPUScatter":
        GPUScatter( int device_id_,
                    int bpg_,

                    int    nQx_,
                    float* h_qx_,
                    int    nQy_,
                    float* h_qy_,
                    int    nQz_,
                    float* h_qz_,

                    int    nAtomsx_,
                    float* h_rx_,
                    int    nAtomsy_,
                    float* h_ry_,
                    int    nAtomsz_,
                    float* h_rz_,
                    int    nAtomsid_,
                    int*   h_id_,

                    int    nCM_,
                    float* h_cm_,

                    int    nRot1_,
                    float* h_rand1_,
                    int    nRot2_,
                    float* h_rand2_,
                    int    nRot3_,
                    float* h_rand3_,

                    int    nQout_,
                    float* h_outQ_
                    ) except +

        
      
cdef class GPUScatter:
        
    cdef C_GPUScatter* thisptr
    
    def __cinit__(self, int device_id, np.ndarray[ndim=3, dtype=np.float32] qxyz,
                  np.ndarray[ndim=3, dtype=np.double_t] rxyz,
                  np.ndarray[ndim=1, dtype=np.int] aid,
                  np.ndarray[ndim=1, dtype=np.double_t] cromermann,
                  ):
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
        del self.thisptr


