import numpy as np
cimport numpy as np
import pylab as plt

from odin import parse
  
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

    #float * polar_pixels    
    void Center(float qMin, float qMax, float center_res, int Nphi_, float size)
    void InterpolateToPolar(float qres_, int Nphi_, int Nq_, float maxq_pix, float maxq, float * polpix)
    
    


# as a reference see:  http://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
# using code borrowed from the stack overflow:
#  http://stackoverflow.com/questions/7543675/how-to-convert-pointer-to-c-array-to-python-array


# I Modded this and added it to the polaripilatus class: 

#np.import_array() # initialize C API to call PyArray_SimpleNewFromData
#cdef public api tonumpyarray(float * data, int size) with gil:
# if not (data and size >= 0): raise ValueError
# cdef np.npy_intp dims = size
# NOTE: it doesn't take ownership of `data`. You must free `data` yourself
# return np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT32, <void*>data)


cdef class polarpilatus:

  cdef PolarPilatus *pp

#    self.pp has a nice ring to it  ~.~

  def __init__(self, cbf_filename,a=0,b=0):
    """
    Converts Pilatus 6M image to polar coordinate image
 
     and a whole lot more ( ^.^ )

    Parameters
    ----------

    cbf_filename : string

    optional:
    a : float, x center on detector (defaults to Xdim/2)
    b : float, y center on detector (defaults to Ydim/2)

    """
    #self.filled_polar=0
    
    cbf     = parse.CBF(cbf_filename)
    vals    = cbf.intensities
    detdist = cbf.path_length
    wavelen = cbf.wavelength
    pixsize = cbf.pixel_size[0]

    X = vals.shape[1]
    Y = vals.shape[0]


    if a == 0 and b == 0:
      a = X/2.
      b = Y/2.
    cdef np.ndarray[ndim=1, dtype=np.float32_t] v
    v = np.ascontiguousarray(vals.flatten(), dtype=np.float32)
    self.pp = new PolarPilatus(int(X), int(Y), &v[0], float(detdist), float(pixsize),
			        float(wavelen), float(a), float(b))
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
 
 
  def center(self,qMin,qMax,center_res=0.5,Nphi=50,size=20.):
    """ finds the center of pilatus image"""
    self.pp.Center(float(qMin),float(qMax),float(center_res),int(Nphi), float(size) )

  def Interpolate_to_polar(self,qres,Nphi):
    """ 
    INTERPOLATES THE PILATUS DETECTOR ADN RETURNS THE RESULT AS A NUMPY ARRAY
    qres :  thickness of each polar ring in inverse angstroms
    Nphi :  number of pixels per polar scattering ring
    """
  
    maxq_pix = np.floor( float (self.pp.Xdim/2.))-15
    if self.pp.Ydim < self.pp.Xdim:
      maxq_pix = np.floor( float (self.pp.Ydim/2.))-15
    Nq=0
    maxq = np.sin( np.arctan2( maxq_pix*self.pp.pixsize, self.pp.detdist ) / 2.)* 4. * np.pi /  self.pp.wavelen
    #for q=0; q < maxq ; q += qres)
    q = 0
    while q < maxq:
      Nq += 1
      q += qres

    #self.filled_polar = 1
    cdef np.ndarray[ndim=1, dtype=np.float32_t] polpix = np.zeros(Nphi*Nq,dtype=np.float32)
    polpix = np.ascontiguousarray(polpix, dtype=np.float32)
    self.pp.InterpolateToPolar(float(qres),int(Nphi),int (Nq) , float (maxq_pix), float( maxq), &polpix[0])
    return polpix.reshape( (Nq,Nphi) )
  
  #def get_polar_pixels(self):
    """ returns the polar pixels as 2d numpy array"""
   # cdef np.npy_intp dims = self.pp.Nphi*self.pp.Nq
    #if self.filled_polar:
    #polpix =  np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT32, <void*> self.pp.polar_pixels)
    #return polpix.reshape( (self.pp.Nq,self.pp.Nphi) )
      ## this is the best I could come up with a.t.m. but im not 100% confident in it
      ## --> just borrowing from the webs.. see the note and refs above
      #polpix = np.fromiter(self.pp.polar_pixels,dtype='float32',count=self.pp.Nphi*self.pp.Nq)
      #print self.pp.Nphi*self.pp.Nq
      #polpix = tonumpyarray(self.pp.polar_pixels,self.pp.Nphi*self.pp.Nq)
      #float * data = self.pp.polar_pixels
    #else:
     # print "Must first form the polar image. Please execute method Interpolate_to_Polar..."

  def I_profile(self,d):
    aves = d.mean(axis=1)
    qvals = [i*self.pp.qres for i in range(len(aves))]
    plt.plot(qvals,aves,linewidth=2)
    plt.xlabel(r'$q\,\AA^{-1}$',fontsize=20)
    plt.ylabel("average intensity",fontsize=20)
    plt.suptitle("Intensity profile",fontsize=20)
    plt.show()
    

