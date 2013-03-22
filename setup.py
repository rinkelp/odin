u"""
setup.py: Install ODIN
"""

import os, sys,re
from os.path import join as pjoin
from glob import glob

#from setuptools import setup, Extension

from distutils.extension import Extension
from distutils.core import setup

from distutils.unixccompiler import UnixCCompiler
from distutils.command.install import install as DistutilsInstall
from Cython.Distutils import build_ext

import numpy

import subprocess
from subprocess import CalledProcessError

# ------------------------------------------------------------------------------
# HEADER
# 

VERSION     = "0.1a"
ISRELEASED  = False
__author__  = "TJ Lane"
__version__ = VERSION

metadata = {
    'name': 'odin',
    'version': VERSION,
    'author': __author__,
    'author_email': 'tjlane@stanford.edu',
    'license': 'GPL v3.0',
    'url': 'https://github.com/tjlane/odin',
    'download_url': 'https://github.com/tjlane/odin',
    'install_requires': ['numpy', 'scipy', 'matplotlib', 'pyyaml', 'mdtraj', 
                         'nose', 'cython>=0.16', 'fabio'],
    'dependency_links' : ['https://github.com/rmcgibbo/mdtraj/tarball/master#egg=mdtraj-0.0.0'],#
                          #'https://fable.svn.sourceforge.net/svnroot/fable/fabio/branches/v0.1.0#egg=fabio-0.1.0'],
    'platforms': ['Linux', 'OSX'],
    'zip_safe': False,
    'test_suite': "nose.collector",
    'description': "Code for Structure Determination",
    'long_description': """ODIN is a simulation toolpackage for producing
models of biomolecular structures consistent with a large set of experimental
data."""}


# ------------------------------------------------------------------------------
# HELPER FUNCTIONS -- path finding, git, python version, readthedocs
# 

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    
    
def print_warning(string):
    print bcolors.WARNING + string + bcolors.ENDC
    

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None
    

# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


def git_version():
    """
    Return the git revision as a string.
    Copied from numpy setup.py
    """
    
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


# ------------------------------------------------------------------------------
# GPU FUNCTION WRAPPING -- nvcc support
# python distutils doesn't have NVCC by default
# 

def locate_cuda():
    """
    Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDA_HOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDA_HOME env variable is in use
    if 'CUDA_HOME' in os.environ:
        home = os.environ['CUDA_HOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            print_warning('The nvcc binary could not be located in your $PATH. '
                          'add it to your path, or set $CUDA_HOME.')
            return False
            
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    print "CUDA config:", cudaconfig
    for k, v in cudaconfig.iteritems():
        if not os.path.exists(v):
            print_warning('The CUDA %s path could not be located in %s' % (k, v))
            return False
    return cudaconfig
    
CUDA = locate_cuda()
if CUDA == False:
    CUDA_SUCCESS = False
else:
    CUDA_SUCCESS = True

def customize_compiler_for_nvcc(self):
    """
    Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """
    
    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

# -----------------------------------------------------------------------------
# INSTALL C/C++ EXTENSIONS
# gpuscatter, cpuscatter, bcinterp
# 


# openmp
if '--no-openmp' in sys.argv[2:]:
    sys.argv.remove('--no-openmp')
    omp_compile = ['-DNO_OMP']
    omp_link    = []
    print_warning('set --no-openmp, disabling OPENMP support')
else:
    omp_compile = ['-fopenmp']
    omp_link    = ['-lgomp']


if CUDA:
    print "Attempting to install GPU functionality"
    gpuscatter = Extension('odin.gpuscatter',
                        sources=['src/scatter/gpuscatter_wrap.pyx', 'src/scatter/gpuscatter.cu'],
                        extra_compile_args={'gcc': ['--fast-math', '-O3', '-fPIC', '-Wall'] + omp_compile,
                                            'g++': ['--fast-math', '-O3', '-fPIC', '-Wall'] + omp_compile,
                                            'nvcc': ['-use_fast_math', '-arch=sm_20', '--ptxas-options=-v', 
                                                     '-c', '--compiler-options', "'-fPIC'"]},
                        library_dirs=[CUDA['lib64']],
                        libraries=['cudart'],
                        runtime_library_dirs=['/usr/lib', '/usr/local/lib', CUDA['lib64']],
                        extra_link_args = ['-lstdc++', '-lm'] + omp_link,
                        include_dirs = [numpy_include, 'src/scatter', CUDA['include']],
                        language='c++')
else:
    gpuscatter = None

cpuscatter = Extension('odin.cpuscatter',
                    sources=['src/scatter/cpuscatter_wrap.pyx', 'src/scatter/cpuscatter.cpp'],
                    extra_compile_args={'gcc': ['--fast-math', '-O3', '-fPIC', '-Wall'] + omp_compile,
                                        'g++': ['--fast-math', '-O3', '-fPIC', '-Wall', '-mmacosx-version-min=10.6'] + omp_compile},
                    runtime_library_dirs=['/usr/lib', '/usr/local/lib'],
                    extra_link_args = ['-lstdc++', '-lm'] + omp_link,
                    include_dirs = [numpy_include, 'src/scatter'],
                    language='c++')
                        
                        
bcinterp = Extension('odin.interp',
                     sources=['src/interp/cyinterp.pyx', 'src/interp/bcinterp.cpp'],
                     extra_compile_args={'gcc': ['--fast-math', '-O3', '-fPIC', '-Wall'] + omp_compile,
                                         'g++': ['--fast-math', '-O3', '-fPIC', '-Wall'] + omp_compile},
                     runtime_library_dirs=['/usr/lib', '/usr/local/lib'],
                     extra_link_args = ['-lstdc++', '-lm'] + omp_link,
                     include_dirs = [numpy_include, 'src/interp'],
                     language='c++')

data = Extension('odin.data',
                     sources=['src/popi/polar_pilatus.pyx', 'src/popi/popi.cpp'],
                     extra_compile_args={'gcc': ['--fast-math', '-O3', '-fPIC', '-Wall'],
                                         'g++': ['--fast-math', '-O3', '-fPIC', '-Wall']},
                     runtime_library_dirs=['/usr/lib', '/usr/local/lib'],
                     extra_link_args = ['-lstdc++', '-lm'],
                     include_dirs = [numpy_include, 'src/popi'],
                     language='c++')

#########################################
#
#            FIND HDF5 DIR
#  (MOST OF THIS STOLEN FROM H5PY)   
#         \\..// ( ^.^ ) \\,,// 
#
#########################################

def scrape_eargs():
    """ Locate settings in environment vars """
    settings = {}

    hdf5 = os.environ.get("HDF5_DIR", "")
    if hdf5 != '':
        settings['hdf5'] = hdf5

    return settings

def scrape_cargs():
    """ Locate settings in command line """
    settings_ = {}
    for arg in sys.argv[:]:
        if arg.find('--hdf5=') == 0:
            hdf5 = arg.split('=')[-1]
            if hdf5.lower() == 'default':
                settings_.pop('hdf5', None)
            else:
                settings_['hdf5'] = hdf5
            sys.argv.remove(arg)
    return settings_

# automatically locates HDF5 install dir if using Enthought python distribution
if re.search('EPD',sys.version):
  HDF5 = sys.prefix                  # default if using EPD
else:
  settings = scrape_eargs()          # lowest priority
  settings.update(scrape_cargs())    # highest priority
  HDF5 = settings.get('hdf5')

if HDF5 is not None:
  ringscatter = Extension('odin.ringscatter',
    sources              = ['src/ring/ring_scatter.pyx','src/ring/ring.cpp'],
    extra_compile_args   = {'gcc':['--fast-math','-O3','-fPIC','-Wall'],
                            'g++':['--fast-math','-O3','-fPIC','-Wall'] },
    include_dirs         = [os.path.join(HDF5, 'include'),'src/ring'],
    library_dirs         = [os.path.join(HDF5, 'lib')],
    runtime_library_dirs = [os.path.join(HDF5, 'lib'),'usr/lib','/usr/local/lib'],
    libraries            = ['hdf5','hdf5_hl'],
    extra_link_args      = ['-lstdc++','-lhdf5'],
    language             =  'c++')
else:
  print "\n  COULD NOT FIND THE HDF5 INSTALL DIRECTORY.\n"
  print "   Please export the environment variable HDF5_DIR\n   or pass the cmd line option --hdf5=/path/to/hdf5"
  print "   to setup.py, where HDF5_DIR or /path/to/hdf5 is\n   the directory containing both the include and lib" 
  print "   directories of your hdf5 build.\n"
  print "   Otherwise proceeding without ringscatter support...\n"

metadata['packages']     = ['odin', 'odin.scripts']
metadata['package_dir']  = {'odin' : 'src/python', 'odin.scripts' : 'scripts'}
metadata['ext_modules']  = [bcinterp, cpuscatter, data]

if gpuscatter:
    metadata['ext_modules'].append(gpuscatter)
    
#if HDF5 is not None: metadata['ext_modules'].append(ringscatter)

metadata['scripts']      = [s for s in glob('scripts/*') if not s.endswith('__.py')]
metadata['data_files']   = [('reference', glob('./reference/*'))]
metadata['cmdclass']     = {'build_ext': custom_build_ext}

# ------------------------------------------------------------------------------
#
# Finally, print a warning at the *end* of the build if something fails
#

def print_warnings():
        
    if not CUDA_SUCCESS:
        print 
        print '*'*65
        print '* WARNING : CUDA/GPU SUPPORT'
        print '* --------------------------'
        print '* Could not install one or more CUDA/GPU features. Look for'
        print '* warnings in the setup.py output (above) for more details. ODIN'
        print '* will function without any GPU-acceleration. Note that for  '
        print '* successful installation of GPU support, you must have an nVidia'
        print '* Fermi-class GPU (or better) and the CUDA toolkit installed. See'
        print '* the nVidia website for more details.'
        print '*'*65
     
    print "\n"

    if HDF5 is None:    
        wrn_str = "*"*65 + "\n"
        wrn_str+= "*  WARNING: COULD NOT FIND THE HDF5 INSTALL DIRECTORY.\n"
        wrn_str+= '*  ---------------------------------------------------\n'
        wrn_str+= "*   Please export the environment variable HDF5_DIR\n*   or pass the cmd line option --hdf5=/path/to/hdf5\n"
        wrn_str+= "*   to setup.py, where HDF5_DIR or /path/to/hdf5 is\n*   the directory containing both the include and lib\n" 
        wrn_str+= "*   directories of your hdf5 build. Careful with env-\n*   variables when using sudo...\n"
        wrn_str+= "*   Otherwise proceeding without ringscatter support...\n"
        wrn_str+= "*"*65
        print wrn_str
    return
        
        
def write_install_py(module_info, filename='src/python/installed.py'):
    """
    module_info is a dict, where each key is an optional module name followed by
    _SUCCESS and each value is a bool indicating if that module was installed
    """

    cnt = """
# THIS FILE IS GENERATED BY ODIN SETUP.PY

# version information
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version

# installed module information
gpuscatter = %(GPUSCATTER_SUCCESS)s
"""

    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    # write `installed.py` which will get installed by distutils
    f = open(filename, 'w')
    try:
        d = {'version': VERSION,
             'full_version' : FULLVERSION,
             'git_revision' : GIT_REVISION,
             'isrelease': str(ISRELEASED)}
        d.update(module_info)
        f.write(cnt % d)
    finally:
        f.close()
        
    return


if __name__ == '__main__':
    write_install_py({'GPUSCATTER_SUCCESS' : CUDA_SUCCESS})
    setup(**metadata) # ** will unpack dictionary 'metadata' providing the values as arguments
    print_warnings()
