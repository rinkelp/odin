
"""
synth.py

Experimental Data & Simulation Synthesis.

This file contains classes necessary for the integration of simulation and
experimental data. Specifically, it serves a Potential class used (by md.py
or mc.py) to run simulations. Further,

"""

class Potential(object):
    
    def __init__(self, expt_data_collection):
        
        pass
        
    def optimize_lambdas(algorithm='default'):
        """
        This is the only place where the experimentally measured data actually 
        gets used.
        """
        pass
        
    def fi(self, xyz):
        """
        Compute all the f_i values for a configuration `xyz`
        """
        pass
        
    def _hessian(self):
        
        pass
        
        
class Predictor(object):
    """
    Abstract base class for predictors of specific obsevables
    """
    
    def f(i, xyz):
        """
        Predict the observable value
        """
        pass
    
    
    def gradf(i, xyz):
        """
        Predict the gradient of the observable
        """
        
    @property
    def error():
        """
        The estimated error associated with the prediction algorithm.
        """
        pass
       
    @property 
    def custom_openmm_fxn():
        """
        If the value of the observable has an analytical form and that form is
        implemented in OpenMM, this property is a string naming that function.
        Otherwise it is `False`.
        """
        return False
        
    @property 
    def has_const_openmm_custom_force():
        """
        True if the value of the obervable can be predicted analytically,
        and that analytical form is readily available, but not implemented in
        OpenMM.
        """
        return False
        
    
        
    
    
class ChemShiftPredictor(Predictor):
    pass
    
class ScatteringPredictor(Predictor):
    pass