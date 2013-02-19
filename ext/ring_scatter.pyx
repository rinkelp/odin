from libcpp.string cimport string

cdef extern from "ring.h":
  cdef cppclass RingScatter:
    RingScatter(string in_file, int Nphi_, int n_rotations_, float qres_, float wavelen_, string qstring_) except + # except + raises python exception in case of c memory allocation error

#cdef RingScatter * rs = new RingScatter("gold_11k.hdf5",360,1,"133 154")
#del  rs



'''
cdef class PyRingScatter:

  cdef RingScatter *thisptr

  def __init__(self,string in_file_, int Nphi__, int n_rotations__, float qres__, float wavelen__ , string qstring__ ):
    self.thisptr = new RingScatter(in_file_, Nphi__, n_rotations__, qres__, wavelen__, qstring__ )
  def __dealloc__ (self):
    del self.thisptr
'''
