from libcpp.string cimport string
from odin.refdata import get_cromermann_parameters
import numpy as np
import tables

cdef extern from "ring.h":
    cdef cppclass RingScatter:
      RingScatter(string in_file, int Nphi_, int n_rotations_, float qres_, float wavelen_, string qstring_) except +

cdef RingScatter * rs

def arr2hdf_simple(h5_file,data_name,data_array,permission):
    f = tables.openFile(h5_file,mode=permission)
    f.createArray(f.root,data_name,data_array)
    f.close()

def params_2_hdf(coor_file, h5_file):
#   open the .coor file and store info as 2D np.array
    xyza     = np.loadtxt(coor_file,np.float32,delimiter=" ")
    atom_ids = xyza[:,3].astype(np.int32)

#   Generate the cromermann data 
#   This code is from cpuscatter_wrap.pyx (tj nice work- love the cromermann grabber)
    cm_param, cm_aid = get_cromermann_parameters(atom_ids)

    rand_posx = np.zeros( n_rotations )
    rand_posy = np.zeros( n_rotations )
    rand_posz = np.zeros( n_rotations )

#   save the coor/cromermann info to h5 using pytables 
    arr2hdf_simple(h5_file, 'xyza',      xyza,      'w')
    arr2hdf_simple(h5_file, 'cm_param',  cm_param,  'a')
    arr2hdf_simple(h5_file, 'cm_aid',    cm_aid,    'a')
    arr2hdf_simple(h5_file, 'rand_posx', rand_posx, 'a')
    arr2hdf_simple(h5_file, 'rand_posy', rand_posy, 'a')
    arr2hdf_simple(h5_file, 'rand_posz', rand_posz, 'a')

def simulate(input_file, Nphi__, n_rotations__, qres__, wavelen__ , qarray, out_file=None):#, output_dir='.' ):   
    """
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
    void : None
        Instead, a file *.ring is created which contains the output of the simultation.
        Use a standars Pytables of hdf5 parser to open the file and explore the contents.
    """ 

    if input_file.split('.')[-1]  == 'coor':
        if out_file == None:
            out_file = input_file.split('.coor')[0]+'.ring'
        params_2_hdf( input_file, out_file, n_rotations )

    elif input_file.split('.')[-1] == 'pdb':
        raise NotImplementedError( 'I still need to add support for .pdb' )

    else:
        raise ValueError( 'The input_file must be either .coor or .pdb' )

#   convert integer list into a string list
#   and then form a space delimited string, 
#   e.g. [133,154,170] --> ['133','154','170'] --> '133 154 170'
    qarray = map( lambda x:str(x), qarray)
    qstring = ' '.join(qarray)

    rs = new RingScatter(out_file, Nphi__, n_rotations__, qres__, wavelen__, qstring )
    
    del rs
