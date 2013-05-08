from libcpp.string cimport string
from odin.refdata import get_cromermann_parameters
import numpy as np
from mdtraj import io
from odin import xray
import tables

cdef extern from "ring.h":
    cdef cppclass RingScatter:
        RingScatter(string in_file, int n_phis_, int n_rotations_, float qres_, float wavelen_, string qstring_, int rands) except +

cdef RingScatter * rs

def arr2hdf_simple( h5_file , data_name , data_array , permission ):
    f = tables.openFile( h5_file , mode=permission )
    f.createArray( f.root , data_name , data_array )
    f.close()

def params_2_hdf(coor_file, h5_file, n, samp_vol, ra ):
#   open the .coor file and store info as 2D np.array
    xyza     = np.loadtxt(coor_file,np.float32,delimiter=" ")
    atom_ids = xyza[:,3].astype(np.int32)

#   Generate the cromermann data 
#   This code is from cpuscatter_wrap.pyx (tj nice work- love the cromermann grabber)
    cm_param, cm_aid = get_cromermann_parameters( atom_ids )

    rand_pos = mol_positions( xyza[:,0:3] , n , samp_vol )

#   save the coor/cromermann info to h5 using pytables 
    arr2hdf_simple(h5_file, 'xyza',     xyza.flatten(),     'w')
    arr2hdf_simple(h5_file, 'cm_param', cm_param.flatten(), 'a')
    arr2hdf_simple(h5_file, 'cm_aid',   cm_aid.flatten(),   'a')
    arr2hdf_simple(h5_file, 'rand_pos', rand_pos,           'a')
    if ra != None:
        if ra.shape != (n , 3 ):
            raise ValueError("If passing random numbers for rotations, please pass a (num_mols x 3) shaped array.")
        arr2hdf_simple(h5_file, 'rands', ra.flatten(), 'a')

def simulate_shot(input_file, n_phis, n_rotations , qres, wavelen , qarray, samp_vol, rands=None, out_file=None): 
    """
    Parameters
    ----------
    input_file : string
        Either a *.coor file or a *.pdb file containing information 
        on the single molecule.
  
    n_phis : int
        Number of azimuthal "pixel" bins around each ring where the 
        scattering is to be computed.
 
    n_rotations : int
        Number of random orientations of the single molecule to sample.
    
    qres : float
        The inverse angstrom unit of the simulation (see qarray description).

    wavelen : float
        The wavelength of the x-ray beam in angstroms.

    qarray : integer list
        A list of integers specifying the rings in q-space at which the 
        scattering is to be computed.The list is provided in units of 
        qres, for instance if n_phis = 360, qres = 0.01, and qarray = [200, 300],
        then simulate will compute the scattering at rings q = 2 A^-1 
        and q = 3 A^-1 , sampling each ring at 360 evenly spaced points.

    samp_vol : float
        The sample volume in microns^3

    Optional Parameters
    -------------------

    rands : np.ndarray float
        Array of numbers on (0-1) used for defining rotations. Should be 
        shape (num_rotations, 3). If rands = None, then the rotations are 
        generated randomly. 
  
    out_file : string
        Name of the output file.
  
    Returns
    -------
    None : void
        Instead, a file *.ring is created which contains the output of the simultation.
        Use a standars Pytables of hdf5 parser to open the file and explore the contents.
    """ 

    if input_file.split('.')[-1]  == 'coor':
        if out_file == None:
            out_file = input_file.split('.coor')[0]+'.ring'
        params_2_hdf( input_file, out_file, n_rotations, samp_vol, rands )

    elif input_file.split('.')[-1] == 'pdb':
        raise NotImplementedError( 'I still need to add support for .pdb' )

    else:
        raise ValueError( 'The input_file must be either .coor or .pdb' )

#   convert integer list into a string list
#   and then form a space delimited string, 
#   e.g. [133,154,170] --> ['133','154','170'] --> '133 154 170'
    qarray = map( lambda x:str(x), qarray)
    qstring = ' '.join(qarray)
    if rands == None:
        rs = new RingScatter(out_file, n_phis, n_rotations, qres, wavelen, qstring, 0 )
    else:
        rs = new RingScatter(out_file, n_phis, n_rotations, qres, wavelen, qstring, 1 )

    del rs


def simulate_shotset(input_file, n_phis, n_rotations , n_shots,  qres, wavelen , qarray, samp_vol, rands=None, out_file=None): 
    """
    Simulate the x-ray scattering onto a polar grid in q-space (scattering vector q ). Produces a file which can be loaded using
    odin.xray.Rings.
     
    Parameters
    ----------
    input_file : string
        Either a *.coor file or a *.pdb file containing information 
        on the single molecule.
  
    n_phis : int
        Number of azimuthal "pixel" bins around each ring where the 
        scattering is to be computed.
 
    n_rotations : int
        Number of random orientations of the single molecule to sample.
    
    n_shots : int
        Number of random orientations of the single molecule to sample.
    
    qres : float
        The inverse angstrom unit of the simulation (see qarray description).

    wavelen : float
        The wavelength of the x-ray beam in angstroms.

    qarray : integer list
        A list of integers specifying the rings in q-space at which the 
        scattering is to be computed.The list is provided in units of 
        qres, for instance if n_phis = 360, qres = 0.01, and qarray = [200, 300],
        then simulate will compute the scattering at rings q = 2 A^-1 
        and q = 3 A^-1 , sampling each ring at 360 evenly spaced points.

    samp_vol : float
        The sample volume in microns^3

    rands : np.ndarray float
        Array of numbers on (0-1) used for defining rotations. Should be 
        shape (num_rotations, 3). If rands = None, then the rotations are 
        generated randomly. 
  
    out_file : string
        Name of the output file.

    Returns
    -------
    None : void
        Instead, an odin Rings file is created which can be opened using xray.Rings.load( *.ring ). 
    """ 

    ofiles = [ 'shot_'+str(i)+'.hdf5' for i in xrange( n_shots ) ]
    keys   = [ ('rings/ringR_'+str(q), 'rings/ringI_'+str(q) ) for q in qarray ]

    n_q    = len(qarray)
    dilu_intens = np.zeros(  (n_shots, n_q, n_phis ) )
    conc_intens = np.zeros(  (n_shots, n_q, n_phis ) )

#   this particular loop should be run across multi procs
    for i_shot in xrange(n_shots ) :
        simulate_shot(input_file, n_phis, n_rotations , qres, wavelen , qarray, samp_vol, rands, ofiles[ i_shot ] )
        
    for i_shot in xrange( n_shots ) :
        F = io.loadh ( ofiles[ i_shot ] )
        
        for i_q in xrange( n_q ) :
            re   = keys[i_q][0]
            im   = keys[i_q][1]

            real = F[ re ] 
            imag = F[ im ] 
            
            dilu = np.sum ( real**2 + imag**2, axis = 0 )
            conc = np.sum ( real,axis=0) **2 + np.sum ( imag,axis=0 ) **2
            
            dilu_intens[  i_shot, i_q, :  ] = dilu
            conc_intens[  i_shot, i_q, :  ] = conc

    qs     = [ qres * q for q in qarray ]
    dilu_R = xray.Rings( qs, dilu_intens, 2*np.pi / wavelen  )
    conc_R = xray.Rings( qs, conc_intens, 2*np.pi / wavelen  )

    dilu_ofile = 'dilu_' + out_file
    conc_ofile = 'conc_' + out_file
    dilu_R.save( dilu_ofile )
    conc_R.save( conc_ofile )


def mol_positions(xyz, n_mols, samp_vol ):
    """
    Disperse molecules randomly in space using this function.
    
    Parameters
    ----------
    xyz : np.ndarray float
        x,y,z positions of atoms in the molecule
        xyz.shape should be (N, 3) where N is the number of atoms in the molecule

    n_mols : int
        number of molecule/particles per shot

    samp_vol : float
        volume of the sample in cubic microns

    Returns
    -------
    ranges : np.array float 
        XYZ positions of center of masses of the particles
    """

    mins = np.min( xyz,axis=0 ) # should be angstrom
    maxs = np.max( xyz,axis=0 ) # should be angstrom

#   this is a box that surrounds the particle
    box_size = np.sqrt ( np.sum( (maxs - mins)**2, axis=0 ) ) / 10. # nm

    samp_dim = np.power( samp_vol, 1./3 ) * 1000. # nm

    rangeX = np.array( [99999999] )
    rangeY = np.array( [99999999] )
    rangeZ = np.array( [99999999] )
    m = 0
    while m < n_mols:
        count = 0
        while 1:
            transX = ( np.random.random() - 0.5 ) * samp_dim
            transY = ( np.random.random() - 0.5 ) * samp_dim
            transZ = ( np.random.random() - 0.5 ) * samp_dim
            if not collisionDetection( rangeX, rangeY, rangeZ,\
                                       transX, transY, transZ, 1.0 * box_size):
                break
            count += 1
            if count == 100000:
                raise RuntimeError('Too many molecules for sample volume.')    
        rangeX = np.append( rangeX, transX )
        rangeY = np.append( rangeY, transY )
        rangeZ = np.append( rangeZ, transZ )
        m += 1

    dens = n_mols * box_size ** 3 / samp_dim**3
    print "The density by volume is",dens,"."
    
    rangeX = np.delete( rangeX,[0])
    rangeY = np.delete( rangeY,[0])
    rangeZ = np.delete( rangeZ,[0])

    ranges = np.vstack( (rangeX,rangeY,rangeZ) ).T.flatten() * 10. #back to angstroms

    return ranges

def collisionDetection(arX,arY,arZ,x,y,z,l):
    collisionX = False
    collisionY = False
    collisionZ = False
    for i in arX:
        if x > i-l and x < i+l:
            collisionX = True
    for i in arY:
        if y > i-l and y < i+l:
            collisionY = True
    for i in arZ:
        if z > i-l and z < i+l:
            collisionZ = True
    if collisionX and collisionY and collisionZ:
        return True
    else:
        return False
