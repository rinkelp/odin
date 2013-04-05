import numpy as np
cimport numpy as np
import pylab as plt

# from odin.arnold import get_arnold as ga

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

    void Center(float qMin, float qMax, float center_res, int Nphi_, float size)
    void InterpolateToPolar(float qres_, int Nphi_, int Nq_, float maxq_pix, float maxq, float * polpix)
    

cdef class polar:
  cdef PolarPilatus *pp

  def __init__(self,d,pixsize,detdist,wavelen,a=0,b=0):
    """
    Converts 2D cartesian detector image into polar coordinates
 
    Parameters
    ----------

    d       : numpy 2D numpy image array 
    pixsize : pixel size in meters
    detdist : samplt -to -detector distance in meters
    wavelen : wavelength of xrays in angstroms

    OPTIONAL PARAMETERS
    ------------------
    a : float, x center on detector (defaults to Xdim/2)
    b : float, y center on detector (defaults to Ydim/2)

    """
    X = d.shape[1]
    Y = d.shape[0]
    if a ==0 and b == 0:
      a = X/2.
      b = Y/2.
  
    cdef np.ndarray[ndim=1, dtype=np.float32_t] v
    v = np.ascontiguousarray(d.flatten(), dtype=np.float32)
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

  def center(self,qMin,qMax,center_res=0.5,Nphi=50,size=20.):
    """ 
    Finds the center of pilatus image:
      polarpilatus.center(qMin,qMax,center_res=0.5,Nphi=50,size=20.)
   
    -------------------
    PARAMS
    ------------------- 
    qMin,qMax  : min and max ring position in pixel units
    center_res : resolution of desired center in pixel units
    Nphi       : number of phi bins when maximizing angular average
    size       : defines a box around the center of the detector (pixel units)
            
    """
    self.pp.Center(float(qMin),float(qMax),float(center_res),int(Nphi), float(size) )

  def Interpolate_to_polar(self,qres,Nphi):
    """ 
    INTERPOLATES THE PILATUS DETECTOR ADN RETURNS THE RESULT AS A NUMPY ARRAY
    qres :  thickness of each polar ring in inverse angstroms
    Nphi :  number of pixels per polar scattering ring
    """
  
    maxq_pix = np.floor( float (self.pp.Xdim/2.))-50
    if self.pp.Ydim < self.pp.Xdim:
      maxq_pix = np.floor( float (self.pp.Ydim/2.))-50
    Nq=0
    maxq = np.sin( np.arctan2( maxq_pix*self.pp.pixsize, self.pp.detdist ) / 2.)* 4. * np.pi /  self.pp.wavelen
    q = 0
    while q < maxq:
      Nq += 1
      q += qres

    cdef np.ndarray[ndim=1, dtype=np.float32_t] polpix = np.zeros(Nphi*Nq,dtype=np.float32)
    polpix = np.ascontiguousarray(polpix, dtype=np.float32)
    self.pp.InterpolateToPolar(float(qres),int(Nphi),int (Nq) , float (maxq_pix), float( maxq), &polpix[0])
    return  polpix.reshape( (Nq,Nphi) )
  
  def I_profile(self,d):
    """
    Plots angular average of scattering pattern, 
    PARAMS: 
    d     : returned numpy array of polarpilatus.Interpolate_to_polar
    """
    aves = d.mean(axis=1)
    qvals = [i*self.pp.qres for i in range(len(aves))]
    plt.plot(qvals,aves,linewidth=2)
    plt.xlabel(r'$q\,\AA^{-1}$',fontsize=20)
    plt.ylabel("average intensity",fontsize=20)
    plt.suptitle("Intensity profile",fontsize=20)
    plt.show()



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
    vals   = cbf.intensities
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
    # print ga()
  
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

  def cartesian_image(self,cbf_filename):
    """ returns a cartesian image of the detector; params: string cbf_filename"""
    cbf     = parse.CBF(cbf_filename)
    vals    = cbf.intensities
    return vals


  def center(self,qMin,qMax,center_res=0.5,Nphi=50,size=20.):
    """ 
    Finds the center of pilatus image:
      polarpilatus.center(qMin,qMax,center_res=0.5,Nphi=50,size=20.)
   
    -------------------
    PARAMS
    ------------------- 
    qMin,qMax  : min and max ring position in pixel units
    center_res : resolution of desired center in pixel units
    Nphi       : number of phi bins when maximizing angular average
    size       : defines a box around the center of the detector (pixel units)
            
    """
    self.pp.Center(float(qMin),float(qMax),float(center_res),int(Nphi), float(size) )

  def Interpolate_to_polar(self,qres,Nphi):
    """ 
    INTERPOLATES THE PILATUS DETECTOR ADN RETURNS THE RESULT AS A NUMPY ARRAY
    qres :  thickness of each polar ring in inverse angstroms
    Nphi :  number of pixels per polar scattering ring
    """
  
    maxq_pix = np.floor( float (self.pp.Xdim/2.))-20
    if self.pp.Ydim < self.pp.Xdim:
      maxq_pix = np.floor( float (self.pp.Ydim/2.))-20
    Nq=0
    maxq = np.sin( np.arctan2( maxq_pix*self.pp.pixsize, self.pp.detdist ) / 2.)* 4. * np.pi /  self.pp.wavelen
    q = 0
    while q < maxq:
      Nq += 1
      q += qres

    cdef np.ndarray[ndim=1, dtype=np.float32_t] polpix = np.zeros(Nphi*Nq,dtype=np.float32)
    polpix = np.ascontiguousarray(polpix, dtype=np.float32)
    self.pp.InterpolateToPolar(float(qres),int(Nphi),int (Nq) , float (maxq_pix), float( maxq), &polpix[0])
    return  polpix.reshape( (Nq,Nphi) )
  
  def I_profile(self,d):
    """
    Plots angular average of scattering pattern, 
    PARAMS: 
    d     : returned numpy array of polarpilatus.Interpolate_to_polar
    """
    aves = d.mean(axis=1)
    qvals = [i*self.pp.qres for i in range(len(aves))]
    plt.plot(qvals,aves,linewidth=2)
    plt.xlabel(r'$q\,\AA^{-1}$',fontsize=20)
    plt.ylabel("average intensity",fontsize=20)
    plt.suptitle("Intensity profile",fontsize=20)
    plt.show()

