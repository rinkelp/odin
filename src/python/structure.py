

"""
structure.py

Functions/classes for manipulating structures.
"""

import numpy as np



class quaternion(object):
    """
    Container class for quaternion-based functions. All methods of this class
    are static, and there is no concept of an instance of this class. It is
    really just a namespace deliminator.
    """
    
    @staticmethod
    def random(rfloat=None):
        """
        Compute a quaterion representing a random rotation, uniform on the
        unit sphere.
        
        Optional Parameters
        -------------------
        rfloat : ndarray, float, len 3
            A 3-vector of random numbers in [0,1) that acts as a random seed. If
            not passed, generates new random numbers.
            
        Returns
        -------
        q : ndarray, float, len 4
            A quaternion representing a random rotation uniform on the unit 
            sphere.
        """
    
        if rfloat == None:
            rfloat = np.random.rand(3)
    
        q = np.zeros(4)
    
        s = rfloat[0]
        sig1 = np.sqrt(s)
        sig2 = np.sqrt(1.0 - s)
    
        theta1 = 2.0 * np.pi * rfloat[1]
        theta2 = 2.0 * np.pi * rfloat[2]
    
        w = np.cos(theta2) * sig2
        x = np.sin(theta1) * sig1
        y = np.cos(theta1) * sig1
        z = np.sin(theta2) * sig2
    
        q[0] = w
        q[1] = x
        q[2] = y
        q[3] = z
    
        return q

    @staticmethod
    def prod(q1, q2):
        """
        Perform the Hamiltonian product of two quaternions. Note that this product
        is non-commutative -- this function returns q1 x q2.
    
        Parameters
        ----------
        q1 : ndarray, float, len(4)
            The first quaternion.
    
        q1 : ndarray, float, len(4)
            The first quaternion.
        
        Returns
        -------
        qprod : ndarray, float, len(4)
            The Hamiltonian product q1 x q2.
        """
        
        if (len(q1) != 4) or (len(q2) != 4):
            raise TypeError('Parameters cannot be interpreted as quaternions')
    
        qprod = np.zeros(4)
    
        qprod[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
        qprod[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
        qprod[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
        qprod[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    
        return qprod
    
    @staticmethod
    def conjugate(q):
        """
        Compute the quaternion conjugate of q.
        
        Parameters
        ----------
        q : ndarray, float, len 4
            A quaternion input.
            
        Returns
        qconj : ndarray, float, len 4
            The conjugate of `q`.
        """
        
        if len(q) != 4:
            raise TypeError('Parameter `q` cannot be interpreted as a quaternion')
    
        qconj = np.zeros(4)
        qconj[0] = q[0]
        qconj[1] = -q[1]
        qconj[2] = -q[2]
        qconj[3] = -q[3]
    
        return qconj

    @staticmethod
    def rand_rotate_vector(v):
        """
        Randomly rotate a three-dimensional vector, `v`, uniformly over the unit
        sphere.
    
        Parameters
        ----------
        v : ndarray, float, len 3
            A vector to rotatea 3-vector in x,y,z space (e.g. the atomic 
            positions of an atom)
            
        Returns
        -------
        v_prime : ndarray, float, len 3
            Another 3-vector, which is the rotated version of v.
        """
        
        if len(v) != 4:
            raise TypeError('Parameter `v` must be in R^3')
        
        # generate a quaternion vector, with the first element zero
        # the last there elements are from v
        qv = np.zeros(4)
        qv[1:] = v.copy()
    
        # get a random quaternion vector
        q = rquaternion()
    
        # take the quaternion conjugated
        qconj = np.zeros(4)
        qconj[0] = q[0]
        qconj[1] = -q[1]
        qconj[2] = -q[2]
        qconj[3] = -q[3]
    
        q_prime = hprod( hprod(q, qv), qconj )
    
        v_prime = q_prime[1:].copy() # want the last 3 elements...
    
        return v_prime


def remove_COM():
    """
    Remove the center of mass from a structure.
    """
    raise NotImplementedError()


def rand_rotate_molecule(xyzlist, remove_COM=False, rfloat=None):
    """
    Randomly rotate the molecule defined by xyzlist.
    
    Parameters
    ----------
    xyzlist : ndarray, float, 3D
        An n x 3 array representing the x,y,z positions of n atoms.
    
    Optional Parameters
    -------------------
    remove_COM : bool
        Whether or not to translate the center of mass of the molecule to the
        origin before rotation.
        
    rfloat : ndarray, float, len 3
        A 3-vector of random numbers in [0,1) that acts as a random seed. If
        not passed, generates new random numbers.
        
    Returns
    -------
    rotated_xyzlist : ndarray, float, 3D
        A rotated version of the input `xyzlist`.
    """
    
    # TJL todo
    if remove_COM:
        raise NotImplementedError()

    # get a random quaternion vector
    q = quaternion.random(rfloat)
    
    # take the quaternion conjugate
    qconj = quaternion.conj(q)
    
    # prepare data structures
    rotated_xyzlist = np.zeros(xyzlist.shape)
    qv = np.zeros(4)
    
    # put each atom through the same rotation
    for i in range(xyzlist.shape[0]):
        qv[1:] = xyzlist[i,:].copy()
        q_prime = quaternion.prod( quaternion.prod(q, qv), qconj )
        rotated_xyzlist[i,:] = q_prime[1:].copy() # want the last 3 elements...
    
    return rotated_xyzlist


def multiply_conformations(xyzlist, num_replicas, concentration):
    """
    Take a structure and generate a template 
    """
    
    
    return