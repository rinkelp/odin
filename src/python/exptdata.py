
"""
exptdata.py

Experimental Data
"""

import os
import abc
from glob import glob

import numpy as np

from mdtraj import io

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


class ExptDataCollection(object):
    """
    A collection of ExptData classes, of various types. This class provides an
    aggregated interface to all those data.
    """
    
    def __init__(self, expt_data_list):
        """
        Generate an instance of the class.
        
        Parameters
        ----------
        expt_data_list : list
            A list of various ExptData instances, mapping to all the data to
            include in this structure prediction round.
        """
        
        self.expt_data_list = expt_data_list
        self._n_data = np.sum([ d._n_data for d in expt_data_list ])
        
        # determine the different kinds of experiments
        self._expttype = []        
        for d in expt_data_list:
            self._expttype.append( type(d).split('.')[-1] ) # dbl chk
        
        return
    
        
    @property
    def n_data(self):
        return self._n_data
    
        
    @property
    def exptmeta(self):
        """
        Metadata necessary for identifying what an experimental value is. Should
        include all information necessary to provide a prediction of that
        experiment, when the x,y,z coordinates of the system are also provided.
        
        E.g., for chemical shifts this would be the atom (index, element type),
        for scattering it would be the detector pixel location (q-vector), etc.
        """
        return self._exptmeta
    
        
    @property
    def expttype(self):
        """
        For each data point, the kind of experiment that generated that data
        """
        assert len(self._expttype) == self._n_data
        return self._expttype
    
        
    @property
    def values(self):
        """
        The measured values of each data point
        """
        
        values = np.zeros(self._n_data)
        
        
        assert values.shape[0] == self._n_data
        return
    
        
    @property
    def errors(self):
        """
        The errors associated with each experiment.
        """
        assert len(self._errors) == self._n_data
        return 
    
        
    def save(self, target):
        """
        Save all experimental data to disk, in one file.
        
        This file needs to contain:
        
        MeasurementID | ExptType | Metadata | Value | Error
        
        """
        
        if os.path.exists(target):
            raise ValueError('File exists! %s' % target)
        
        if target.endswith('.db'):
            logger.info('Writing expt. data to database: %s' % target)
            self._to_sqlite(target)
        elif target.endswith('.hdf'):
            logger.info('Writing expt. data to HDF5: %s' % target)
            self._to_file(target)
        else:
            raise ValueError('Cannot understand what format to write %s to. Options: .hdf, .db.' % target)

    @classmethod
    def load(self, target):
        pass
    

class ExptData(object):
    """
    Abstract base class for experimental data classes.
        
    All ExptData objects should have the following properties:
    -- values (the experimental values)
    -- errors (the STD error associated with values)
        
    Further, each ExptData inheretant should provide a self.predict(xyz) method,
    that outputs an array of len(values) that is the *prediction* of the array
    `values` given a molecular geometry `xyz`.
    """
    
    __metaclass__ = abc.ABCMeta
        
    # the following methods will be automatically inhereted, but are mutable
    # by children if need be    
            
    @property
    def n_data(self):
        return self._n_data
        
    @property
    def values(self):
        """
        The measured values of each data point
        """
        return self._get_values()
        
    @property
    def errors(self):
        """
        The errors associated with each experiment.
        """
        return self._get_errors()

    # Classes that inherent from ExptData must implement all the methods below 
    # this is enforced by the abstract class module
    
    @abc.abstractmethod
    def from_file(self, filename):
        """
        Load a file containing experimental data, and dump the relevant data,
        errors, and metadata into the object in a smart/comprehendable manner.
        """
        return
        
    @abc.abstractmethod
    def predict(self, trajectory):
        """
        Method to predict the array `values` for each snapshot in `trajectory`.
        
        Parameters
        ----------
        trajectory : mdtraj.trajectory
            A trajectory to predict the experimental values for.
        
        Returns
        -------
        prediction : ndarray, 2-D
           The predicted values. Will be two dimensional, 
           len(trajectory) X len(values).
        """
        return prediction
        
    @abc.abstractmethod
    def _default_error(self):
        """
        Method to estimate the error of the experiment (conservatively) in the
        absence of explicit input.
        """
        return error_guess
        
    @abc.abstractmethod
    def _get_values(self):
        """
        Return an array `values`, in an order that ensures it will match up
        with the method self.predict()
        """
        return values

    @abc.abstractmethod
    def _get_errors(self):
        """
        Return an array `errors`, in an order that ensures it will match up
        with the method self.predict()
        """
        return errors
    

class DistanceRestraint(ExptData):
    """
    An experimental data class that can be used for NOEs, chemical x-linking,
    etc. Anything that relies on a distance restraint.
    
    The input here is an N x 4 array, with each row specifying
    
    row     entry
    ---     -----
     1  --> atom index 1
     2  --> atom index 2
     3  --> distance between atoms 1 & 2, in Angstroms
     4  --> restraint is satisfied in experiment or not (0 or 1)
        
    The function then returns a "1" if the restraint is satisfied, a "0" 
    otherwise. Note that by setting an experimental restraint that is not
    satisfied, you are saying the distance must be *greater* than the indicated
    value.
    """
        
    def __init__(self, restraint_array, errors=None):
        """
        Instantiate a DistanceRestraint experimental data class.
        
        An experimental data class that can be used for NOEs, chemical x-linking,
        etc. Anything that relies on a distance restraint.

        Parameters
        ----------
        restraint_array : np.ndarray, float
            An array in the following format:
            
                row     entry
                ---     -----
                 1  --> atom index 1
                 2  --> atom index 2
                 3  --> distance between atoms 1 & 2, in Angstroms
                 4  --> restraint is satisfied in experiment or not (0 or 1)

            The function then returns a "1" if the restraint is satisfied, a "0" 
            otherwise. Note that by setting an experimental restraint that is 
            not satisfied, you are saying the distance must be *greater* than  
            the indicated value. 
            
            Actually, so long as the error parameter is not set to zero, these
            can take fractional values representing the confidence with which
            they can be assigned.
            
        Optional Parameters
        -------------------
        errors : np.ndarray, float
            An array of the standard deviations representing the confidence in
            each restraint.
        """
        
        if not type(restraint_array) == np.ndarray:
            raise TypeError('`restraint_array` must be type np.ndarray, got '
                            '%s' % type(restraint_array))
        
        if not ((restraint_array.shape[1] == 4) and (len(restraint_array.shape) == 2)):
            raise ValueError('`restraint_array` must have shape (n,4), got '
                             '%s' % str(restraint_array.shape))
        
        self.restraint_array = restraint_array
        self._n_data = restraint_array.shape[0]
        
        if errors == None:
            self._errors = self._default_error()
        else:
            self._errors = errors
        
        return
    
        
    @classmethod
    def from_file(cls, filename):
        """
        Load a file containing experimental data, and dump the relevant data,
        errors, and metadata into the object in a smart/comprehendable manner.
        """
                          
        if filename.endswith('dat'):
            restraint_array = np.loadtxt(filename)
        else:
            raise IOError('cannot understand ext of %s, must be one of: '
                          '%s' % (filename, self.acceptable_filetypes))
        
        return cls(restraint_array)
    
        
    def predict(self, trajectory):
        """
        Method to predict the array `values` for each snapshot in `trajectory`.
        In this case, `values` is an array of 0's (distance bigger than
        restraint) and 1's (distance smaller than restraint)
        
        Parameters
        ----------
        trajectory : mdtraj.trajectory
            A trajectory to predict the experimental values for.
        
        Returns
        -------
        prediction : ndarray, 2-D
           The predicted values. Will be two dimensional, 
           len(trajectory) X len(values).
        """
        
        prediction = np.zeros( (trajectory.n_frames, self._n_data),
                                dtype=self.restraint_array.dtype )
        
        for i in range(trajectory.n_frames):
            for j in range(self._n_data):
                
                ai = int(self.restraint_array[j,0]) # index 1
                aj = int(self.restraint_array[j,1]) # index 2
                
                # compute the atom-atom dist and compare to the input (dist in ang)
                d = np.sqrt( np.sum( np.power(trajectory.xyz[i,ai,:] - \
                                              trajectory.xyz[i,aj,:], 2) ) ) * 10.0
                
                logger.debug('distance from atom %d to %d: %f A' % (ai, aj, d))
                                              
                if d > self.restraint_array[j,2]:
                    prediction[i,j] = 0.0
                else:
                    prediction[i,j] = 1.0
        
        return prediction
    
        
    def _default_error(self):
        """
        Method to estimate the error of the experiment (conservatively) in the
        absence of explicit input.
        
        The errors are standard deviations around the mean, and thus are an
        one-D ndarray of len(values).
        """
        # special case -- since this expt value is binary
        return np.ones(self._n_data) * 0.1 # todo think of something better
    
        
    def _get_values(self):
        """
        Return an array `values`, in an order that ensures it will match up
        with the method self.predict()
        """
        values = self.restraint_array[:,3] # binary values
        return values
    
        
    def _get_errors(self):
        """
        Return an array `errors`, in an order that ensures it will match up
        with the method self.predict()
        """
        # this method is dumb for this class, see if we need it for others
        return self._errors
