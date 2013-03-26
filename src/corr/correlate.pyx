import numpy as np
cimport numpy as np
import random

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

def randPairs(numItems,numPairs):
  """
  generates random pairs of integers
  Parameters
  ----------
    numItems: int
    numPairs: int
  Returns
  -------
    list of ints [[i1,j1], [i2,j2], ... [iN,jN]]
  """
  random.seed()
  i = 0
  pairs = []
  while i < numPairs:
    ind1 = random.randrange(numItems)
    ind2 = ind1
    while ind2 == ind1:
      ind2 = random.randrange(numItems)
    pair = [ind1,ind2]
    if pairs.count(pair) == 0:
      pairs.append(pair)
      i += 1
  return pairs

def intra(ringsA,ringsB,num_cors=0):
  """
  does intra shot correlations for many shots..
  PARAMS ringsA  : ndarray floats (first dim = number of rings, second dim = number of pixels per ring) 
         ringsB  : ndarray floats (first dim = number of rings, second dim = number of pixels per ring) 
  OPT    num_cors:  int number of corrs to compute
  RETURNS 1-d numpy array (average correlation)  
  """
  if num_cors == 0:
    num_cors = ringsA.shape[0]
  intra = np.zeros(( num_cors,ringsA.shape[1] ))
  for i in xrange(num_cors):
    intra[i] = correlate(ringsA[i],ringsB[i])
  return intra

def inter(ringsA,ringsB,num_cors = 0):
  """
  does inter shot correlations for many shots..
  PARAMS ringsA  : ndarray floats (first dim = number of rings, second dim = number of pixels per ring) 
         ringsB  : ndarray floats (first dim = number of rings, second dim = number of pixels per ring) 
  OPT    num_cors:  int number of corrs to compute
  RETURNS 1-d numpy array (average correlation)  
  """
  if num_cors == 0:
    num_cors = ringsA.shape[0]
  inter = np.zeros( (num_cors, ringsA.shape[1] ))
  pairs = randPairs(ringsA.shape[1],num_cors)
  k = 0
  for i,j in pairs:
    inter[k] = correlate(ringsA[i],ringsB[j])
    k += 1
  return inter

