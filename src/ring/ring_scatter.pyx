
cdef extern from "ring.h":
  cdef cppclass RingScatter:
    RingScatter(string, int, int, float, float, string) except + # except + raises python exception in case of c memory allocation error

#cdef RingScatter * rs = new RingScatter("gold_11k.hdf5",360,1,"133 154")
#del  rs

cdef class PyRingScatter:

  cdef RingScatter *thisptr

  def __init__(self,string in_file, int Nphi_, int n_rotations_, float qres_, float wavelen_ , string qstring_ ):
    self.thisptr = new RingScatter(in_file, Nphi_, n_rotations_, qres_, wavlen_, qstring_ )
  def __dealloc__ (self):
    del self.thisptr
