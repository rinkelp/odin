import numpy as np
cimport numpy as np
import random
from scipy import fftpack

cdef extern from "corr.h":
  cdef cppclass Corr:
    Corr(int N_, float * ar1, float * ar2, float * ar3) except +

cdef Corr * c

def correlate(A,B):
  """
  compute the correlation between 2 arrays
  Parameters:
  -----------
  A, 1D numpy array
  B, 1D numpy array 
  """
  if A.shape != B.shape:
    print "arrays must be of same size and shape" 
    return 0  # will add an traceback once I learn how :)
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

def correlate_using_fft(self, x, y):
  """
  Compute the correlation between 2 arrays using the 
  convolution theorem. Works well for unmasked arrays.
  Passing masked arrays will results in numerical errors.

  Parameters
  ----------
  x : 1d numpy array of floats
      The intensities along ring 1
  y : 1d numpy array of floats
      The intensities along ring 2
    
  Returns
  -------
  iff : 1d numpy darray, float
      The correlation between x and y
  """

  xmean = x.mean()
  ymean = y.mean()

  # use d-FFT + convolution thm
  ffx = fftpack.fft( x-xmean )
  ffy = fftpack.fft( y-ymean )
  iff = np.real( fftpack.ifft( np.conjugate(ffx) * ffy ) ) / xmean  / ymean

  return iff / float ( x.shape[0] )


