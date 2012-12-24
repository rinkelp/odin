
import os
import subprocess
from nose import SkipTest

from mdtraj import trajectory

from odin import xray
from odin.testing import skip, ref_file

MPL = True
try:
    import matplotlib
except:
    MPL = False

try:
    from odin import gpuscatter
    GPU = True
except ImportError as e:
    GPU = False
    

class TestShoot(object):
    
    def setup(self):
        self.file = ref_file('ala2.pdb')
        
    def test_single_gpu(self):
        if not GPU: raise SkipTest
        cmd = 'shoot -s %s -n 1 -m 512 -o testshot.shot > /dev/null 2>&1' % self.file
        subprocess.check_call(cmd, shell=True)
        if not os.path.exists('testshot.shot'):
            raise RuntimeError('no output produced')
        else:
            s = xray.Shot.load('testshot.shot')
            os.remove('testshot.shot')
            
    def test_cpu(self):
        cmd = 'shoot -s %s -n 1 -m 1 -o testshot2.shot > /dev/null 2>&1' % self.file
        subprocess.check_call(cmd, shell=True)
        if not os.path.exists('testshot2.shot'):
            raise RuntimeError('no output produced')
        else:
            s = xray.Shot.load('testshot2.shot')
            os.remove('testshot2.shot')
        
        
def test_plotiq():
    if not MPL: raise SkipTest
    cmd = 'plotiq -i %s -m 1.0 > /dev/null 2>&1' % ref_file('refshot.shot')
    subprocess.check_call(cmd, shell=True)
    if not os.path.exists('intensity_plot.pdf'):
        raise RuntimeError('no output produced')
    else:
        os.remove('intensity_plot.pdf')
        
        
def test_plotcorr():
    if not MPL: raise SkipTest
    cmd = 'plotcorr -i %s > /dev/null 2>&1' % ref_file('refshot.shot')
    subprocess.check_call(cmd, shell=True)
    if not os.path.exists('correlation_plot.pdf'):
        raise RuntimeError('no output produced')
    else:
        os.remove('correlation_plot.pdf')
        
        
def test_replicate():
    cmd = 'replicate -i %s -n 10 -d 0.1 > /dev/null 2>&1' % ref_file('goldBenchMark.coor')
    subprocess.check_call(cmd, shell=True)
    if not os.path.exists('replicated.pdb'):
        raise RuntimeError('no output produced')
    else:
        o = trajectory.load('replicated.pdb')
        os.remove('replicated.pdb')
        
