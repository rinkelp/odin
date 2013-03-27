
"""
synth.py

Experimental Data & Simulation Synthesis.

This file contains classes necessary for the integration of simulation and
experimental data. Specifically, it serves a Potential class used (by md.py
or mc.py) to run simulations. Further,

"""

class ExptPotential(object):
    """
    A posterior potential that enforces a set of experimental constraints.
    """
    
    def __init__(self, expt_data_collection):
        
        pass
        
        
    def __call__(self, xyz):
        """
        Takes a set of xyz coordinates and evaluates the potential on that
        conformation.
        
        """
        
        return energy
    
        
    def prior(self):
        # need to think!
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
