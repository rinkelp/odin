from libcpp.string cimport string

from odin.refdata import get_cromermann_parameters

import numpy as np

import tables

from odin.arnold import get_arnold as ga


cdef extern from "ring.h":
  cdef cppclass RingScatter:
    RingScatter(string in_file, int Nphi_, int n_rotations_, float qres_, float wavelen_, string qstring_) except + # except + raises python exception in case of c memory allocation error

cdef RingScatter * rs #= new RingScatter("gold_11k.hdf5",360,1,"133 154")

def arr2hdf_simple(h5_file,data_name,data_array,permission):
  f = tables.openFile(h5_file,mode=permission)
  f.createArray(f.root,data_name,data_array)
  f.close()

def coor_and_CM_2_hdf(coor_file, h5_file):
# open the .coor file and store info as 2D np.array
  f        = open(coor_file,'r')
  lines    = f.readlines()
  xyza     = np.zeros( (len(lines),4) ,dtype='float32')
  atom_ids = np.zeros( len(lines),dtype='int32')
  for i in xrange( len(lines) ):
    line = lines[i].strip().split()
    x = float( line[0] )
    y = float( line[1] )
    z = float( line[2] )
    a = float( line[3] )
    xyza[i,:]   = [x,y,z,a]
    atom_ids[i] = int( line[3] )
  f.close()
# Generate the cromermann data 
# This code is from cpuscatter_wrap.pyx (tj nice work- love the cromermann grabber)
  cm_param, cm_aid = get_cromermann_parameters(atom_ids) 

# save the coor/cromermann info to h5 using pytables 
  arr2hdf_simple(h5_file,'xyza',    xyza,    'w')
  arr2hdf_simple(h5_file,'cm_param',cm_param,'a')
  arr2hdf_simple(h5_file,'cm_aid',  cm_aid,  'a')

def simulate(input_file, Nphi__, n_rotations__, qres__, wavelen__ , qarray):#, output_dir='.' ):   
  '''
  parameters
  ----------
  input_file : string
    Either a *.coor file or a *.pdb file containing information on the single molecule
  
  Nphi__ : int
    Number of azimuthal "pixel" bins around each ring where the scattering is to be computed
 
  n_rotations__ : int
    Number of random orientations of the single molecule to sample

  qres__ : float
    The inverse angstrom unit of the simulation (see qarray description)

  wavelen__ : float
    The wavelength of the x-ray beam in angstroms

  qarray : integer list
    A list of integers specifying the rings in q-space at which the scattering is to be computed.
    The list is provided in units of qres, for instance if Nphi = 360, qres = 0.01, and qarray = [200, 300],
    then simulate will compute the scattering at rings q = 2 A^-1 and q = 3 A^-1 , sampling each ring at 
    360 evenly spaced points.
  

  Returns
  -------
  None; Instead, a file *.ring is created which contains the output of the simultation.
  Use a standars Pytables of hdf5 parser to open the file and explore the contents.
  ''' 
  if input_file.split('.')[-1]  == 'coor':
    in_file_ = input_file.split('.')[0]+'.ring'
    coor_and_CM_2_hdf( input_file, in_file_ )
  elif input_file.split('.')[-1] == 'pdb':
    'Still need to add support for .pdb'
    return 0
  else:
    print " input_file Must be either .coor or .pdb. Exiting..."
    return 0

# convert integer list into a string list
# and then form a space delimited string, 
# e.g. [133,154,170] --> ['133','154','170'] --> '133 154 170'
  qarray = map( lambda x:str(x), qarray)
  qstring = ' '.join(qarray)

  rs = new RingScatter(in_file_, Nphi__, n_rotations__, qres__, wavelen__, qstring )
  del rs
  print ga()
