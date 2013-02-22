import numpy as np
cimport numpy as np

  
cdef extern from "popi.h":
  cdef cppclass PolarPilatus:

    PolarPilatus(int   Xdim_,     int   Ydim_,     float * binData,
            	 float detdist_,  float pixsize_,  float wavelen_, 
	         float x_center_, float y_center_) except +

    float pixsize  # size of square pixel in meters
    float detdist  # sample to detector distance in meters
    float wavelen  # wavelength of xray beam in angstroms
    float x_center # beam center in x direction
    float y_center # beam center in y direction
    float qres     # resolution of polar-converted image in inverse angstroms
    int   Xdim     # X dimension of detector
    int   Ydim     # Y dimension of detector
    int   Nq       # Radial dimension of polar-coverted detector
    int   Nphi     # Azimuthal dimension of polar-converted detector

    float * polar_pixels    
    void Center(float qMin, float qMax, float center_res, int Nphi_, float size)
    void InterpolateToPolar(float qres_, int Nphi_)


cdef class polar_pilatus:

  cdef PolarPilatus *pp

#    self.pp has a nice ring to it  ~.~

  def __init__(self, vals, detdist, pixsize,wavelen,a=0,b=0):
    """
    Converts Pilatus 6M image to polar coordinate image
 
     and a whole lot more ( ^.^ )

    Parameters
    ----------

    """
    X = int(vals.shape[1])
    Y = int(vals.shape[0])

    if a == 0 and b == 0:
      a = float(X/2.)
      b = float(Y/2.)
    cdef np.ndarray[ndim=1, dtype=np.float32_t] v
    v = np.ascontiguousarray(vals.flatten(), dtype=np.float32)
    self.pp = new PolarPilatus(X, Y, &v[0], detdist, pixsize,
			        wavelen, a, b)
  def __dealloc__(self):
    del self.pp
  
  property Xdim:
    def __get__(self): 
      return self.pp.Xdim
  property Ydim:
    def __get__(self): 
      return self.pp.Ydim
  property Nphi:
    def __get__(self): 
      return self.pp.Nphi
  property Nq:
    def __get__(self): 
      return self.pp.Nq
  property x_center:
    def __get__(self): 
      return self.pp.x_center
  property y_center:
    def __get__(self): 
      return self.pp.y_center
  property wavelen:
    def __get__(self): 
      return self.pp.wavelen
  property pixsize:
    def __get__(self): 
      return self.pp.pixsize
  property detdist:
    def __get__(self): 
      return self.pp.detdist
  property qres:
    def __get__(self): 
      return self.pp.qres
  #property polar_pixels:
  #  def __get__(self):
  #    return self.polar_pixels[0]
  
  def center(self,qMin,qMax,center_res,Nphi,size):
    """ finds the center of pilatus image"""
    self.pp.Center(float(qMin),float(qMax),float(center_res),int(Nphi), float(size) )

  def Interpolate_to_polar(self,qres,Nphi):
    self.pp.InterpolateToPolar(float(qres),int(Nphi))

