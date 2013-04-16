
import numpy as np
cimport numpy as np

cdef extern from "corr.h":
  cdef cppclass Corr:
    Corr(int N_, float * ar1, float * ar2, float * ar3) except +

cdef Corr * c

def correlate(A, B):
    """
    compute the correlation between 2 arrays. If any element of the array is
    zero, then treat that value as a "mask" -- i.e. ignore it.
    
    Parameters
    ----------
    A, 1D numpy array
    B, 1D numpy array 
    """
    
    if A.shape != B.shape:
        raise ValueError("arrays A, B must be of same size/shape")

    N = A.shape[0]
    C = np.zeros_like(A)
    
    cdef np.ndarray[ndim=1,dtype=np.float32_t] v1
    cdef np.ndarray[ndim=1,dtype=np.float32_t] v2
    cdef np.ndarray[ndim=1,dtype=np.float32_t] v3

    v1 = np.ascontiguousarray(A.flatten(),dtype=np.float32)
    v2 = np.ascontiguousarray(B.flatten(),dtype=np.float32)
    v3 = np.ascontiguousarray(C.flatten(),dtype=np.float32)
    c  = new Corr(N,&v1[0], &v2[0], &v3[0])
    del c
    return v3


