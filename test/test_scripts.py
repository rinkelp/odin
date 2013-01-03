
import os
import subprocess
from nose import SkipTest

from mdtraj import trajectory

from odin import xray
from odin.testing import skip, ref_file


try:
    import matplotlib
    MPL = True
except:
    MPL = False

try:
    from odin import gpuscatter
    GPU = True
except ImportError as e:
    GPU = False


try:
    import openmm
    OPENMM = True
except ImportError as e:
    OPENMM = False
    
# see if we are on Travis CI -- which for whatever reason does not play well with 
# these tests that use subprocess. todo : find out why and see if we can fix it
try:
    if os.environ['TRAVIS'] == 'true':
        TRAVIS = True
    else:
        TRAVIS = False
except:
    TRAVIS = False
    

class TestShoot(object):
    
    def setup(self):
        self.file = ref_file('ala2.pdb')
        
    def test_single_gpu(self):
        if not GPU: raise SkipTest
        if TRAVIS: raise SkipTest
        cmd = 'shoot -s %s -n 1 -m 512 -o testshot.shot > /dev/null 2>&1' % self.file
        subprocess.check_call(cmd, shell=True)
        if not os.path.exists('testshot.shot'):
            raise RuntimeError('no output produced')
        else:
            s = xray.Shot.load('testshot.shot')
            os.remove('testshot.shot')
            
    def test_cpu(self):
        if TRAVIS: raise SkipTest
        cmd = 'shoot -s %s -n 1 -m 1 -o testshot2.shot > /dev/null 2>&1' % self.file
        subprocess.check_call(cmd, shell=True)
        if not os.path.exists('testshot2.shot'):
            raise RuntimeError('no output produced')
        else:
            s = xray.Shot.load('testshot2.shot')
            os.remove('testshot2.shot')
        
        
def test_plotiq():
    if not MPL: raise SkipTest
    if TRAVIS: raise SkipTest
    cmd = 'plotiq -i %s -m 1.0 > /dev/null 2>&1' % ref_file('refshot.shot')
    subprocess.check_call(cmd, shell=True)
    if not os.path.exists('intensity_plot.pdf'):
        raise RuntimeError('no output produced')
    else:
        os.remove('intensity_plot.pdf')
        
        
def test_plotcorr():
    if not MPL: raise SkipTest
    if TRAVIS: raise SkipTest
    cmd = 'plotcorr -i %s > /dev/null 2>&1' % ref_file('refshot.shot')
    subprocess.check_call(cmd, shell=True)
    if not os.path.exists('correlation_plot.pdf'):
        raise RuntimeError('no output produced')
    else:
        os.remove('correlation_plot.pdf')
        
        
def test_replicate():
    if TRAVIS: raise SkipTest
    cmd = 'replicate -i %s -n 10 -d 0.1 > /dev/null 2>&1' % ref_file('goldBenchMark.coor')
    subprocess.check_call(cmd, shell=True)
    if not os.path.exists('replicated.pdb'):
        raise RuntimeError('no output produced')
    else:
        o = trajectory.load('replicated.pdb')
        os.remove('replicated.pdb')
        
        
def test_solvate():
    if not OPENMM: raise SkipTest
    if TRAVIS: raise SkipTest
    cmd = 'solvate -i %s > /dev/null 2>&1' % ref_file('ala2.pdb')
    subprocess.check_call(cmd, shell=True)
    if not os.path.exists('solvated.pdb'):
        raise RuntimeError('no output produced')
    else:
        o = trajectory.load('solvated.pdb')
        os.remove('solvated.pdb')
        
        
def test_cbf2shot():
    if TRAVIS: raise SkipTest
    cmd = 'cbf2shot -i %s -o test.shot > /dev/null 2>&1' % ref_file('test1.cbf')
    subprocess.check_call(cmd, shell=True)
    if not os.path.exists('test.shot'):
        raise RuntimeError('no output produced')
    else:
        o = xray.Shotset.load('test.shot')
        os.remove('test.shot')
        
