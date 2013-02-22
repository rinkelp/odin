import numpy as np
cimport numpy as np

  
cdef extern from "popi.hh":
  cdef cppclass Polar_Pilatus "PolarPilatus":

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
    


      
cdef class PolarPilatus:
        
    cdef Polar_Pilatus * pp

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
        self.pp = new Polar_Pilatus(X, Y, &v[0], detdist, pixsize,
                                   wavelen, a, b)
    def __dealloc__(self):
        del self.pp
    
    def __get__(self): 
        return self.thisptr.x0
    
    def __set__(self, x0):
        self.thisptr.x0 = x0

    def center(qMin,qMax,center,center_res,Nphi,size):
	""" finds the center of pilatus image"""
        self.pp.Center(float(qMin),float(qMax),float(center_res),int(Nphi), float(size) )

    def Interpolate_to_polar(qres,Nphi):
        self.pp.InterpolateToPolar(float(qres),int(Nphi))
