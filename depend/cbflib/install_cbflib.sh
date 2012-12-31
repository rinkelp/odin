#!/usr/bin/bash
# ---------------------------------------------------------------------------- #
#
# A simple script that attempts to install
# (1) cbflib
# (2) pycbf
#
# Should this (rather simple and dumb) script fail, attempt a manual 
# installation. Either edit one of the makefiles in this directory, or look in
# ./more_makefiles for the makefile that best matches your system, and
# edit that. Then you should be able to 
#
#   $ make all
#   $ make install
#
# install in any location you desire.
#
# Next, attempt to install pycbf. You'll need SWIG, but if you have it you
# should be in the clear. Just do the following:
#
#   $ cd pycbf
#   $ python setup.py install # may need to be root
#
# and make sure everything worked
#
#   $ python -c "import pycbf"
#
# run the ODIN tests to make sure... that will provide a much more thorough test
# of both pycbf and cbflib itself. Good luck.
#
# written for use with ODIN by TJL, 12.23.12
#
# ---------------------------------------------------------------------------- #


# choose the makefile appropriate for the system
SYSTEM=`uname -a | cut -d " " -f 1`

if [ $SYSTEM == "Darwin" ]
then
    echo "Installing cbflib & pycbf for OSX"
    MAKEFILE="Makefile_OSX"
elif [ $SYSTEM == "Linux" ]
then
    echo "Installing cbflib & pycbf for LINUX"
    MAKEFILE="Makefile_LINUX"
else
    echo "Installing cbflib & pycbf -- COULD NOT DETECT OS TYPE!"
    MAKEFILE="Makefile_Default"
fi
    
ln -s $MAKEFILE Makefile

# this script will install in ./build
mkdir build

# indicate we don't want FORTRAN or long long types
export CBF_DONT_USE_LONG_LONG=1
export NOFORTRAN=1

# attempt the make
make all
make install

# install pycbf
cd pycbf
python setup.py install

