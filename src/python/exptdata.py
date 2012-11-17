
"""
exptdata.py

Experimental Data
"""

import os
from glob import glob
import numpy as np

from mdtraj import io


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
        
        self._n_data = np.sum([ d._n_data for d in expt_data_list ])
        self._directories = [d._directory for d in expt_data_list]
        
        self._files = {}
        self._expttype = []
        self._exptmeta = []
        values         = []
        errors         = []
        for d in expt_data_list:
            self._files[d._directory] = d._files
            
            if isinstance(d, ScatteringData):
                self._expttype.append('scattering')
            elif isinstance(d, ChemShiftData):
                self._expttype.append('chemshift')
            else:
                raise RuntimeError('Invalid experimental data type. Currently'
                                   'allowed: ScatteringData, ChemShiftData.')
                                   
            self._exptmeta.append(d._exptmeta)
            
            values.append(d._values)
            errors.append(d._errors)
            
        self._values = np.array(values)
        self._errors = np.array(errors)
        
        assert(len(self._values) == self._n_data)
        assert(len(self._errors) == self._n_data)
        assert(len(self._exptmeta) == self._n_data)

    @property
    def n_data(self):
        return self._n_data

    @property
    def directories(self):
        return self._directories
        
    @property
    def files(self):
        return self._files
        
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
        return self._expttype
        
    @property
    def values(self):
        """
        The measured values of each data point
        """
        return self._values
        
    @property
    def errors(self):
        """
        The errors associated with each experiment.
        """
        return self._errors
        

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

    def load(self, target):
        pass
        
    def _to_sqlite(self, dbfilename):
        pass
        
    def _from_sqlite(self, dbfilename):
        pass
        
    def _to_file(self, filemame):
        # use Cpickle, binary serialization
    
    @classmethod
    def _from_file(self, filename):
        pass


class ExptData(object):
    """
    Abstract base class for experimental data classes
    """
    
    def __init__(self, directory):
        """
        Initialize the class, loading into memory all experimental data.
        
        
        """
        
        self._filetypes = ''
        self._directory = directory
        
        self._files = []
        for filetype in self._filetypes:
            self._files += glob(directory + '/*' + filetype)
            
            
        self._values = []
        self._errors = []
        self._metadata = []
        for fn in self._files:
            values, errors, metadata = self.load(fn)
            self._values.append(values)
            self._errors.append(errors)
            self._errors.append(metadata)
        self._values = np.array(values)
        self._errors = np.array(errors)
        
        self._n_data = len(self._values)
        
        assert(len(self._errors) == self._n_data)
        assert(len(self._metadata) == self._n_data)
        
    
    def n_data(self):
        return self._n_data
    
    @property
    def directory(self):
        return self._directory
        
    @property
    def files(self):
        return self._files
        
    @property
    def values(self):
        """
        The measured values of each data point
        """
        return self._values

    @property
    def errors(self):
        """
        The errors associated with each experiment.
        """
        return self._errors

    # TJL : need to figure out how ABC's work. All classes should have below methods
    @abstractplaceholder
    def _load_file(self, filename):
        return values, errors, metadata
        
    @abstractplaceholder
    def _default_error():
        """ estimates the error of the experiment (conservatively) in the
            absence of explicit input  """
        return error_guess        


class ScatteringData(ExptData):
    
    def _load_file(self, filename):
        return values, errors, metadata
        
    def _default_error():
        """ estimates the error of the experiment (conservatively) in the
            absence of explicit input  """
        return error_guess
        
    
class ChemShiftData(ExptData):
    
    def _load_file(self, filename):
        return values, errors, metadata
        
    def _default_error():
        """ estimates the error of the experiment (conservatively) in the
            absence of explicit input  """
        return error_guess
