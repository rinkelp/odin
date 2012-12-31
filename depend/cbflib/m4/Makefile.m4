m4_define(`cbf_version',`0.9.2')m4_dnl
m4_define(`cbf_date',`12 Feb 2011')m4_dnl
m4_ifelse(cbf_system,`',`m4_define(`cbf_system',`LINUX')')
`######################################################################
#  Makefile - command file for make to create CBFlib                 #
#                                                                    #
# Version 'cbf_version cbf_date`                                          #
#                                                                    #
#                          Paul Ellis and                            #
#         Herbert J. Bernstein (yaya@bernstein-plus-sons.com)        #
#                                                                    #
# (C) Copyright 2006 - 2011 Herbert J. Bernstein                     #
#                                                                    #
######################################################################

######################################################################
#                                                                    #
# YOU MAY REDISTRIBUTE THE CBFLIB PACKAGE UNDER THE TERMS OF THE GPL #
#                                                                    #
# ALTERNATIVELY YOU MAY REDISTRIBUTE THE CBFLIB API UNDER THE TERMS  #
# OF THE LGPL                                                        #
#                                                                    #
######################################################################

########################### GPL NOTICES ##############################
#                                                                    #
# This program is free software; you can redistribute it and/or      #
# modify it under the terms of the GNU General Public License as     #
# published by the Free Software Foundation; either version 2 of     #
# (the License, or (at your option) any later version.               #
#                                                                    #
# This program is distributed in the hope that it will be useful,    #
# but WITHOUT ANY WARRANTY; without even the implied warranty of     #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the      #
# GNU General Public License for more details.                       #
#                                                                    #
# You should have received a copy of the GNU General Public License  #
# along with this program; if not, write to the Free Software        #
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA           #
# 02111-1307  USA                                                    #
#                                                                    #
######################################################################

######################### LGPL NOTICES ###############################
#                                                                    #
# This library is free software; you can redistribute it and/or      #
# modify it under the terms of the GNU Lesser General Public         #
# License as published by the Free Software Foundation; either       #
# version 2.1 of the License, or (at your option) any later version. #
#                                                                    #
# This library is distributed in the hope that it will be useful,    #
# but WITHOUT ANY WARRANTY; without even the implied warranty of     #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU  #
# Lesser General Public License for more details.                    #
#                                                                    #
# You should have received a copy of the GNU Lesser General Public   #
# License along with this library; if not, write to the Free         #
# Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston,    #
# MA  02110-1301  USA                                                #
#                                                                    #
######################################################################

######################################################################
#                                                                    #
#                    Stanford University Notices                     #
#  for the CBFlib software package that incorporates SLAC software   #
#                 on which copyright is disclaimed                   #
#                                                                    #
# This software                                                      #
# -------------                                                      #
# The term "this software", as used in these Notices, refers to      #
# those portions of the software package CBFlib that were created by #
# employees of the Stanford Linear Accelerator Center, Stanford      #
# University.                                                        #
#                                                                    #
# Stanford disclaimer of copyright                                   #
# --------------------------------                                   #
# Stanford University, owner of the copyright, hereby disclaims its  #
# copyright and all other rights in this software.  Hence, anyone    #
# may freely use it for any purpose without restriction.             #
#                                                                    #
# Acknowledgement of sponsorship                                     #
# ------------------------------                                     #
# This software was produced by the Stanford Linear Accelerator      #
# Center, Stanford University, under Contract DE-AC03-76SFO0515 with #
# the Department of Energy.                                          #
#                                                                    #
# Government disclaimer of liability                                 #
# ----------------------------------                                 #
# Neither the United States nor the United States Department of      #
# Energy, nor any of their employees, makes any warranty, express or #
# implied, or assumes any legal liability or responsibility for the  #
# accuracy, completeness, or usefulness of any data, apparatus,      #
# product, or process disclosed, or represents that its use would    #
# not infringe privately owned rights.                               #
#                                                                    #
# Stanford disclaimer of liability                                   #
# --------------------------------                                   #
# Stanford University makes no representations or warranties,        #
# express or implied, nor assumes any liability for the use of this  #
# software.                                                          #
#                                                                    #
# Maintenance of notices                                             #
# ----------------------                                             #
# In the interest of clarity regarding the origin and status of this #
# software, this and all the preceding Stanford University notices   #
# are to remain affixed to any copy or derivative of this software   #
# made or distributed by the recipient and are to be affixed to any  #
# copy of software made or distributed by the recipient that         #
# contains a copy or derivative of this software.                    #
#                                                                    #
# Based on SLAC Software Notices, Set 4                              #
# OTT.002a, 2004 FEB 03                                              #
######################################################################



######################################################################
#                               NOTICE                               #
# Creative endeavors depend on the lively exchange of ideas. There   #
# are laws and customs which establish rights and responsibilities   #
# for authors and the users of what authors create.  This notice     #
# is not intended to prevent you from using the software and         #
# documents in this package, but to ensure that there are no         #
# misunderstandings about terms and conditions of such use.          #
#                                                                    #
# Please read the following notice carefully.  If you do not         #
# understand any portion of this notice, please seek appropriate     #
# professional legal advice before making use of the software and    #
# documents included in this software package.  In addition to       #
# whatever other steps you may be obliged to take to respect the     #
# intellectual property rights of the various parties involved, if   #
# you do make use of the software and documents in this package,     #
# please give credit where credit is due by citing this package,     #
# its authors and the URL or other source from which you obtained    #
# it, or equivalent primary references in the literature with the    #
# same authors.                                                      #
#                                                                    #
# Some of the software and documents included within this software   #
# package are the intellectual property of various parties, and      #
# placement in this package does not in any way imply that any       #
# such rights have in any way been waived or diminished.             #
#                                                                    #
# With respect to any software or documents for which a copyright    #
# exists, ALL RIGHTS ARE RESERVED TO THE OWNERS OF SUCH COPYRIGHT.   #
#                                                                    #
# Even though the authors of the various documents and software      #
# found here have made a good faith effort to ensure that the        #
# documents are correct and that the software performs according     #
# to its documentation, and we would greatly appreciate hearing of   #
# any problems you may encounter, the programs and documents any     #
# files created by the programs are provided **AS IS** without any   *
# warranty as to correctness, merchantability or fitness for any     #
# particular or general use.                                         #
#                                                                    #
# THE RESPONSIBILITY FOR ANY ADVERSE CONSEQUENCES FROM THE USE OF    #
# PROGRAMS OR DOCUMENTS OR ANY FILE OR FILES CREATED BY USE OF THE   #
# PROGRAMS OR DOCUMENTS LIES SOLELY WITH THE USERS OF THE PROGRAMS   #
# OR DOCUMENTS OR FILE OR FILES AND NOT WITH AUTHORS OF THE          #
# PROGRAMS OR DOCUMENTS.                                             #
######################################################################

######################################################################
#                                                                    #
#                           The IUCr Policy                          #
#      for the Protection and the Promotion of the STAR File and     #
#     CIF Standards for Exchanging and Archiving Electronic Data     #
#                                                                    #
# Overview                                                           #
#                                                                    #
# The Crystallographic Information File (CIF)[1] is a standard for   #
# information interchange promulgated by the International Union of  #
# Crystallography (IUCr). CIF (Hall, Allen & Brown, 1991) is the     #
# recommended method for submitting publications to Acta             #
# Crystallographica Section C and reports of crystal structure       #
# determinations to other sections of Acta Crystallographica         #
# and many other journals. The syntax of a CIF is a subset of the    #
# more general STAR File[2] format. The CIF and STAR File approaches #
# are used increasingly in the structural sciences for data exchange #
# and archiving, and are having a significant influence on these     #
# activities in other fields.                                        #
#                                                                    #
# Statement of intent                                                #
#                                                                    #
# The IUCr''`s interest in the STAR File is as a general data          #
# interchange standard for science, and its interest in the CIF,     #
# a conformant derivative of the STAR File, is as a concise data     #
# exchange and archival standard for crystallography and structural  #
# science.                                                           #
#                                                                    #
# Protection of the standards                                        #
#                                                                    #
# To protect the STAR File and the CIF as standards for              #
# interchanging and archiving electronic data, the IUCr, on behalf   #
# of the scientific community,                                       #
#                                                                    #
# # holds the copyrights on the standards themselves,                *
#                                                                    #
# # owns the associated trademarks and service marks, and            *
#                                                                    #
# # holds a patent on the STAR File.                                 *
#                                                                    #
# These intellectual property rights relate solely to the            #
# interchange formats, not to the data contained therein, nor to     #
# the software used in the generation, access or manipulation of     #
# the data.                                                          #
#                                                                    #
# Promotion of the standards                                         #
#                                                                    #
# The sole requirement that the IUCr, in its protective role,        #
# imposes on software purporting to process STAR File or CIF data    #
# is that the following conditions be met prior to sale or           #
# distribution.                                                      #
#                                                                    #
# # Software claiming to read files written to either the STAR       *
# File or the CIF standard must be able to extract the pertinent     #
# data from a file conformant to the STAR File syntax, or the CIF    #
# syntax, respectively.                                              #
#                                                                    #
# # Software claiming to write files in either the STAR File, or     *
# the CIF, standard must produce files that are conformant to the    #
# STAR File syntax, or the CIF syntax, respectively.                 #
#                                                                    #
# # Software claiming to read definitions from a specific data       *
# dictionary approved by the IUCr must be able to extract any        #
# pertinent definition which is conformant to the dictionary         #
# definition language (DDL)[3] associated with that dictionary.      #
#                                                                    #
# The IUCr, through its Committee on CIF Standards, will assist      #
# any developer to verify that software meets these conformance      #
# conditions.                                                        #
#                                                                    #
# Glossary of terms                                                  #
#                                                                    #
# [1] CIF:  is a data file conformant to the file syntax defined     #
# at http://www.iucr.org/iucr-top/cif/spec/index.html                #
#                                                                    #
# [2] STAR File:  is a data file conformant to the file syntax       #
# defined at http://www.iucr.org/iucr-top/cif/spec/star/index.html   #
#                                                                    #
# [3] DDL:  is a language used in a data dictionary to define data   #
# items in terms of "attributes". Dictionaries currently approved    #
# by the IUCr, and the DDL versions used to construct these          #
# dictionaries, are listed at                                        #
# http://www.iucr.org/iucr-top/cif/spec/ddl/index.html               #
#                                                                    #
# Last modified: 30 September 2000                                   #
#                                                                    #
# IUCr Policy Copyright (C) 2000 International Union of              #
# Crystallography                                                    #
######################################################################


# Version string
VERSION = 'cbf_version`


#
# Comment out the next line if scratch test files sould be retain
#
CLEANTESTS = yes

'm4_ifelse(cbf_use_pycifrw,`yes',`
#
# Definitions to get versions of PyCifRW and PLY
#
PYCIFRW = PyCifRW-3.3_6Dec09
PLY = ply-3.2
PYCIFRWFLAG = -DCBF_USE_PYCIFRW
')m4_dnl
`
#
# Definition to get a version of tifflib to support tiff2cbf
#
TIFF = tiff-3.9.4-rev-6Feb11
TIFFPREFIX = $(PWD)


#
# Definitions to get a stable version of regex
#
REGEX = regex-20090805
REGEXDIR = /usr/lib
REGEXDEP = 

# Program to use to retrieve a URL

DOWNLOAD = wget

# Flag to control symlinks versus copying

SLFLAGS = --use_ln

#
# Program to use to pack shars
#
SHAR	= /usr/bin/shar
#SHAR	= /usr/local/bin/gshar

#
# Program to use to create archives
#
AR	= /usr/bin/ar

#
# Program to use to add an index to an archive
#
RANLIB  = /usr/bin/ranlib

#
# Program to use to decompress a data file
#
DECOMPRESS = /usr/bin/bunzip2


#
# Program to use to compress a data file
#
COMPRESS = /usr/bin/bzip2

#
# Program to use to generate a signature
#
SIGNATURE = /usr/bin/openssl dgst -md5


#
# Extension for compressed data file (with period)
#
CEXT = .bz2

#
# Extension for signatures of files
#
SEXT = .md5


# call to time a command

#TIME =
#TIME = time

#
# Program to display differences between files
#
DIFF = diff -u -b


#
# Program to generate wrapper classes for Python
#
PYSWIG = swig -python

#
# Program to generate wrapper classes for Java
#
JSWIG = swig -java

#
# Program to generate LaTex and HTML program documentation
#

NUWEB = nuweb

#
# Compiler for Java
#
JAVAC = javac

#
# Java archiver for compiled classes
#
JAR = jar

#
# Java SDK root directory
#
ifeq ($(JDKDIR),)
  JDKDIR 	=	/usr/lib/java
endif

ifneq ($(CBF_DONT_USE_LONG_LONG),)
NOLLFLAG = -DCBF_DONT_USE_LONG_LONG
else
NOLLFLAG =
endif

#
# PYCBF definitions
#
PYCBFEXT = so
PYCBFBOPT =
SETUP_PY = setup.py

#
# Set the compiler and flags
#
'm4_ifelse(cbf_system,`OSX',`
#########################################################
#
#  Appropriate compiler definitions for MAC OS X
#  Also change defintion of DOWNLOAD
#
#########################################################
CC	= gcc
C++	= g++
CFLAGS  = -g -O2  -Wall -ansi -pedantic
F90C = gfortran
F90FLAGS = -g -fno-range-check
F90LDFLAGS = -bind_at_load
EXTRALIBS = -lm
M4FLAGS = -Dfcb_bytes_in_rec=131072
TIME = time
DOWNLOAD = /sw/bin/wget',
cbf_system,`OSX_gcc42',`
#########################################################
#
#  Appropriate compiler definitions for MAC OS X
#  with gcc 4.2
#  Also change defintion of DOWNLOAD
#
#########################################################
CC	= gcc
C++	= g++
CFLAGS  = -g -O2  -Wall -ansi -pedantic
F90C = gfortran
F90FLAGS = -g -fno-range-check
F90LDFLAGS = -bind_at_load
SOCFLAGS = -fPIC
SOLDFLAGS = -shared -Wl,-rpath,$(INSTALLDIR)/lib
JAVAINCLUDES = -I$(JDKDIR)/include -I$(JDKDIR)/include/linux
LDPREFIX = LD_LIBRARY_PATH=$(SOLIB)
EXTRALIBS = -lm
M4FLAGS = -Dfcb_bytes_in_rec=131072
TIME = time
DOWNLOAD = /sw/bin/wget',
cbf_system,`OSX_gcc42_DMALLOC',`
#########################################################
#
#  Appropriate compiler definitions for MAC OS X
#  with gcc 4.2 and DMALLOC
#  Also change defintion of DOWNLOAD
#
#########################################################
CC	= gcc
C++	= g++
CFLAGS  = -g -O2  -Wall -ansi -pedantic -DDMALLOC -DDMALLOC_FUNC_CHECK -I$(HOME)/include
F90C = gfortran
F90FLAGS = -g -fno-range-check
F90LDFLAGS = -bind_at_load
SOCFLAGS = -fPIC
SOLDFLAGS = -shared -Wl,-rpath,$(INSTALLDIR)/lib
JAVAINCLUDES = -I$(JDKDIR)/include -I$(JDKDIR)/include/linux
LDPREFIX = LD_LIBRARY_PATH=$(SOLIB)
EXTRALIBS = -lm -L$(HOME)/lib -ldmalloc
M4FLAGS = -Dfcb_bytes_in_rec=131072
TIME = time
DOWNLOAD = /sw/bin/wget',
cbf_system,`LINUX_64',`
#########################################################
#
#  Appropriate compiler definitions for Linux x86_64
#  with gcc version 4.2
#
#########################################################
CC	= gcc -m64
C++	= g++ -m64
CFLAGS  = -g -O2 -Wall -D_USE_XOPEN_EXTENDED -fno-strict-aliasing
F90C = gfortran -m64
F90FLAGS = -g -fno-range-check
F90LDFLAGS =
SOCFLAGS = -fPIC
SOLDFLAGS = -shared -Wl,-rpath,$(INSTALLDIR)/lib
JAVAINCLUDES = -I$(JDKDIR)/include -I$(JDKDIR)/include/linux
LDPREFIX = LD_LIBRARY_PATH=$(SOLIB)
EXTRALIBS = -lm
M4FLAGS = -Dfcb_bytes_in_rec=131072
TIME = time
DOWNLOAD = /sw/bin/wget',
cbf_system,`LINUX_gcc42',`
#########################################################
#
#  Appropriate compiler definitions for Linux
#  with gcc version 4.2
#
#########################################################
CC	= gcc
C++	= g++
CFLAGS  = -g -O2 -Wall -D_USE_XOPEN_EXTENDED -fno-strict-aliasing
F90C = gfortran
F90FLAGS = -g -fno-range-check
F90LDFLAGS = 
SOCFLAGS = -fPIC
SOLDFLAGS = -shared -Wl,-rpath,$(INSTALLDIR)/lib
JAVAINCLUDES = -I$(JDKDIR)/include -I$(JDKDIR)/include/linux
LDPREFIX = LD_LIBRARY_PATH=$(SOLIB)
EXTRALIBS = -lm
M4FLAGS = -Dfcb_bytes_in_rec=131072
TIME = time',
cbf_system,`LINUX',`
#########################################################
#
#  Appropriate compiler definitions for Linux
#
#########################################################
CC	= gcc
C++	= g++
CFLAGS  = -g -O2 -Wall -D_USE_XOPEN_EXTENDED -fno-strict-aliasing
F90C = gfortran
F90FLAGS = -g
F90LDFLAGS = 
SOCFLAGS = -fPIC
SOLDFLAGS = -shared -Wl,-rpath,$(INSTALLDIR)/lib
JAVAINCLUDES = -I$(JDKDIR)/include -I$(JDKDIR)/include/linux
LDPREFIX = LD_LIBRARY_PATH=$(SOLIB)
EXTRALIBS = -lm
M4FLAGS = -Dfcb_bytes_in_rec=131072
TIME = time',
cbf_system,`LINUX_gcc42_DMALLOC',`
#########################################################
#
#  Appropriate compiler definitions for Linux
#  with gcc version 4.2 and DMALLOC
#
#########################################################
CC	= gcc
C++	= g++
CFLAGS  = -g -O2 -Wall -D_USE_XOPEN_EXTENDED -fno-strict-aliasing -DDMALLOC -DDMALLOC_FUNC_CHECK  -I$(HOME)/include
F90C = gfortran
F90FLAGS = -g -fno-range-check
F90LDFLAGS = 
SOCFLAGS = -fPIC
SOLDFLAGS = -shared -Wl,-rpath,$(INSTALLDIR)/lib
JAVAINCLUDES = -I$(JDKDIR)/include -I$(JDKDIR)/include/linux
LDPREFIX = LD_LIBRARY_PATH=$(SOLIB)
EXTRALIBS = -lm -L$(HOME)/lib -ldmalloc
M4FLAGS = -Dfcb_bytes_in_rec=131072
TIME = time',
cbf_system,`LINUX_DMALLOC',`
#########################################################
#
#  Appropriate compiler definitions for Linux and DMALLOC
#
#########################################################
CC	= gcc
C++	= g++
CFLAGS  = -g -O2 -Wall -D_USE_XOPEN_EXTENDED -fno-strict-aliasing -DDMALLOC -DDMALLOC_FUNC_CHECK  -I$(HOME)/include
F90C = gfortran
F90FLAGS = -g
F90LDFLAGS = 
SOCFLAGS = -fPIC
SOLDFLAGS = -shared -Wl,-rpath,$(INSTALLDIR)/lib
JAVAINCLUDES = -I$(JDKDIR)/include -I$(JDKDIR)/include/linux
LDPREFIX = LD_LIBRARY_PATH=$(SOLIB)
EXTRALIBS = -lm -L$(HOME)/lib -ldmalloc
M4FLAGS = -Dfcb_bytes_in_rec=131072
TIME = time',
cbf_system,`AIX',`
#########################################################
#
#  Appropriate compiler definitions for AIX
#
#########################################################
CC	= xlc
C++	= xlC
CFLAGS  = -g -O2  -Wall
F90C = xlf90
F90FLAGS = -g -qsuffix=f=f90
F90LDFLAGS = 
M4FLAGS = -Dfcb_bytes_in_rec=131072
EXTRALIBS = -lm
TIME = time',
cbf_system,`MINGW',`
#########################################################
#
#  Appropriate compiler definitions for Mingw
#  Also change from symlinks to copies and
#  use default paths for utilities
#
#########################################################
CC	= gcc
C++	= g++
CFLAGS  = -g -O2  -Wall -static -I/usr/include -fno-strict-aliasing
F90C = g95
F90FLAGS = -g
F90LDFLAGS = 
M4FLAGS = -Dfcb_bytes_in_rec=4096
SOCFLAGS = -D_JNI_IMPLEMENTATION_
SOLDFLAGS = -shared -Wl,--kill-at
JAVAINCLUDES = -I$(JDKDIR)/include -I$(JDKDIR)/include/win32
EXTRALIBS = -L$(REGEXDIR) -lregex -lm
REGEXDEP = $(REGEXDIR)/libregex.a
TIME =
PYCBFEXT = pyd
PYCBFBOPT = --compiler=mingw32
SETUP_PY = setup_MINGW.py
JDKDIR = /java
JSWIG = /swig/swig -java
PYSWIG = /swig/swig -python
SLFLAGS = --use_cp
SHAR	= shar
AR		=  ar
RANLIB  =  ranlib
DECOMPRESS =  bunzip2',
cbf_system,`IRIX_gcc',`
#########################################################
#
#  Appropriate compiler definitions for IRIX w. gcc
#  No f90
#  use default paths for utilities
#
#########################################################
CC      = gcc
C++     = g++
CFLAGS  = -g -O2  -Wall
F90C    =
F90FLAGS =
M4FLAGS = -Dfcb_bytes_in_rec=4096
EXTRALIBS = -lm
TIME    =
SHAR    = shar
AR      =  ar
RANLIB  =
DECOMPRESS =  bunzip2',

`
#########################################################
#
#  Appropriate compiler definitions for default (Linux)
#
#########################################################
CC	= gcc
C++	= g++
CFLAGS  = -g -O2 -Wall -D_USE_XOPEN_EXTENDED -fno-strict-aliasing
F90C = gfortran
F90FLAGS = -g -fno-range-check
F90LDFLAGS = 
SOCFLAGS = -fPIC
SOLDFLAGS = -shared -Wl,-rpath,$(INSTALLDIR)/lib
JAVAINCLUDES = -I$(JDKDIR)/include -I$(JDKDIR)/include/linux
LDPREFIX = LD_LIBRARY_PATH=$(SOLIB)
EXTRALIBS = -lm
M4FLAGS = -Dfcb_bytes_in_rec=131072
TIME = time')`

ifneq ($(NOFORTRAN),)
F90C =
endif

#
# Directories
#
ROOT     = .
LIB      = $(ROOT)/lib
SOLIB    = $(ROOT)/solib
JCBF     = $(ROOT)/jcbf
JAVADIR  = $(ROOT)/java
BIN      = $(ROOT)/bin
SRC      = $(ROOT)/src
INCLUDE  = $(ROOT)/include
M4       = $(ROOT)/m4
PYCBF    = $(ROOT)/pycbf
EXAMPLES = $(ROOT)/examples
DECTRIS_EXAMPLES = $(EXAMPLES)/dectris_cbf_template_test
DOC      = $(ROOT)/doc
GRAPHICS = $(ROOT)/html_graphics
DATADIRI  = $(ROOT)/../CBFlib_$(VERSION)_Data_Files_Input
DATADIRO  = $(ROOT)/../CBFlib_$(VERSION)_Data_Files_Output
DATADIRS  = $(ROOT)/../CBFlib_$(VERSION)_Data_Files_Output_Sigs_Only
INSTALLDIR  = $(HOME)

#
# URLs from which to retrieve the data directories
#
DATAURLBASE	= http://downloads.sf.net/cbflib/
DATAURLI	= $(DATAURLBASE)/CBFlib_$(VERSION)_Data_Files_Input.tar.gz
DATAURLO	= $(DATAURLBASE)/CBFlib_$(VERSION)_Data_Files_Output.tar.gz
DATAURLS	= $(DATAURLBASE)/CBFlib_$(VERSION)_Data_Files_Output_Sigs_Only.tar.gz

#
# URLs from which to retrieve needed external package snapshots
#
'm4_ifelse(cbf_use_pycifrw,`yes',`
PYCIFRWURL	= http://downloads.sf.net/cbflib/$(PYCIFRW).tar.gz
PLYURL		= http://www.dabeaz.com/ply/$(PLY).tar.gz
')m4_dnl
`REGEXURL	= http://downloads.sf.net/cbflib/$(REGEX).tar.gz
TIFFURL		= http://downloads.sf.net/cbflib/$(TIFF).tar.gz


#
# Include directories
#
INCLUDES = -I$(INCLUDE) -I$(SRC)


######################################################################
#  You should not need to make modifications below this line         #
######################################################################


#
# Suffixes of files to be used or built
#
.SUFFIXES:	.c .o .f90 .m4

.m4.f90:
		m4 -P $(M4FLAGS) $< > $@

ifneq ($(F90C),)
.f90.o:
		$(F90C) $(F90FLAGS) -c $< -o $@
endif


#
# Common dependencies
#
COMMONDEP = Makefile

#
# Source files
#
SOURCE   =  $(SRC)/cbf.c               \
			$(SRC)/cbf_alloc.c         \
			$(SRC)/cbf_ascii.c         \
			$(SRC)/cbf_binary.c        \
			$(SRC)/cbf_byte_offset.c   \
			$(SRC)/cbf_canonical.c     \
			$(SRC)/cbf_codes.c         \
			$(SRC)/cbf_compress.c      \
			$(SRC)/cbf_context.c       \
			$(SRC)/cbf_copy.c          \
			$(SRC)/cbf_file.c          \
			$(SRC)/cbf_getopt.c        \
			$(SRC)/cbf_lex.c           \
			$(SRC)/cbf_packed.c        \
			$(SRC)/cbf_predictor.c     \
			$(SRC)/cbf_read_binary.c   \
			$(SRC)/cbf_read_mime.c     \
			$(SRC)/cbf_simple.c        \
			$(SRC)/cbf_string.c        \
			$(SRC)/cbf_stx.c           \
			$(SRC)/cbf_tree.c          \
			$(SRC)/cbf_uncompressed.c  \
			$(SRC)/cbf_write.c         \
			$(SRC)/cbf_write_binary.c  \
			$(SRC)/cbf_ws.c            \
			$(SRC)/md5c.c

'm4_ifelse(cbf_use_pycifrw,`yes',`			
PYSOURCE  = $(SRC)/drel_lex.py		   \
			$(SRC)/drel_yacc.py		   \
			$(SRC)/drelc.py \
			$(SRC)/drel_prep.py
')m4_dnl
`
F90SOURCE = $(SRC)/fcb_atol_wcnt.f90     \
			$(SRC)/fcb_ci_strncmparr.f90 \
			$(SRC)/fcb_exit_binary.f90   \
			$(SRC)/fcb_nblen_array.f90   \
			$(SRC)/fcb_next_binary.f90   \
			$(SRC)/fcb_open_cifin.f90    \
			$(SRC)/fcb_packed.f90        \
			$(SRC)/fcb_read_bits.f90     \
			$(SRC)/fcb_read_byte.f90     \
			$(SRC)/fcb_read_image.f90    \
			$(SRC)/fcb_read_line.f90     \
			$(SRC)/fcb_read_xds_i2.f90   \
			$(SRC)/fcb_skip_whitespace.f90 \
			$(EXAMPLES)/test_fcb_read_image.f90 \
			$(EXAMPLES)/test_xds_binary.f90
			
		   
#
# Header files
#
HEADERS   =  $(INCLUDE)/cbf.h                  \
			 $(INCLUDE)/cbf_alloc.h            \
			 $(INCLUDE)/cbf_ascii.h            \
			 $(INCLUDE)/cbf_binary.h           \
			 $(INCLUDE)/cbf_byte_offset.h      \
			 $(INCLUDE)/cbf_canonical.h        \
			 $(INCLUDE)/cbf_codes.h            \
			 $(INCLUDE)/cbf_compress.h         \
			 $(INCLUDE)/cbf_context.h          \
			 $(INCLUDE)/cbf_copy.h             \
			 $(INCLUDE)/cbf_file.h             \
			 $(INCLUDE)/cbf_getopt.h           \
			 $(INCLUDE)/cbf_lex.h              \
			 $(INCLUDE)/cbf_packed.h           \
			 $(INCLUDE)/cbf_predictor.h        \
			 $(INCLUDE)/cbf_read_binary.h      \
			 $(INCLUDE)/cbf_read_mime.h        \
			 $(INCLUDE)/cbf_simple.h           \
			 $(INCLUDE)/cbf_string.h           \
			 $(INCLUDE)/cbf_stx.h              \
			 $(INCLUDE)/cbf_tree.h             \
			 $(INCLUDE)/cbf_uncompressed.h     \
			 $(INCLUDE)/cbf_write.h            \
			 $(INCLUDE)/cbf_write_binary.h     \
			 $(INCLUDE)/cbf_ws.h               \
			 $(INCLUDE)/global.h               \
			 $(INCLUDE)/cbff.h                 \
			 $(INCLUDE)/md5.h

#
# m4 macro files
#
M4FILES   = $(M4)/fcblib_defines.m4            \
			$(M4)/fcb_exit_binary.m4           \
			$(M4)/fcb_next_binary.m4           \
			$(M4)/fcb_open_cifin.m4            \
			$(M4)/fcb_packed.m4                \
			$(M4)/fcb_read_bits.m4             \
			$(M4)/fcb_read_image.m4            \
			$(M4)/fcb_read_xds_i2.m4           \
			$(M4)/test_fcb_read_image.m4       \
			$(M4)/test_xds_binary.m4


#
# Documentation files
#
DOCUMENTS = $(DOC)/CBFlib.html                   \
			$(DOC)/CBFlib.txt                   \
			$(DOC)/CBFlib_NOTICES.html          \
			$(DOC)/CBFlib_NOTICES.txt           \
			$(DOC)/ChangeLog                    \
			$(DOC)/ChangeLog.html               \
			$(DOC)/MANIFEST                     \
			$(DOC)/gpl.txt $(DOC)/lgpl.txt

#
# HTML Graphics files
#
JPEGS     = $(GRAPHICS)/CBFbackground.jpg      \
			$(GRAPHICS)/CBFbig.jpg             \
			$(GRAPHICS)/CBFbutton.jpg          \
			$(GRAPHICS)/cbflibbackground.jpg   \
			$(GRAPHICS)/cbflibbig.jpg          \
			$(GRAPHICS)/cbflibbutton.jpg       \
			$(GRAPHICS)/cifhome.jpg            \
			$(GRAPHICS)/iucrhome.jpg           \
			$(GRAPHICS)/noticeButton.jpg


#
# Default: instructions
#
default:
	@echo ''` ''`
	@echo ''`***************************************************************''`
	@echo ''` ''`
	@echo ''` PLEASE READ README and doc/CBFlib_NOTICES.txt''`
	@echo ''` ''`
	@echo ''` Before making the CBF library and example programs, check''`
	@echo ''` that the C compiler name and flags are correct:''`
	@echo ''` ''`
	@echo ''` The current values are:''`
	@echo ''` ''`
	@echo ''`   $(CC) $(CFLAGS) $(NOLLFLAG) $(PYCIFRWFLAG)''`
	@echo ''` ''`
	@echo ''` Before installing the CBF library and example programs, check''`
	@echo ''` that the install directory is correct:''`
	@echo ''` ''`
	@echo ''` The current value :''`
	@echo ''` ''`
	@echo ''`   $(INSTALLDIR) ''`	
	@echo ''` ''`
	@echo ''` To compile the CBF library and example programs type:''`
	@echo ''` ''`
	@echo ''`   make clean''`
	@echo ''`   make all''`
	@echo ''` ''`
	@echo ''` To compile the CBF library as a shared object library, type:''`
	@echo ''` ''`
	@echo ''`   make shared''`
	@echo ''` ''`
	@echo ''` To compile the Java wrapper classes for CBF library, type:''`
	@echo ''` ''`
	@echo ''`   make javawrapper''`
	@echo ''` ''`
	@echo ''` To run a set of tests type:''`
	@echo ''` ''`
	@echo ''`   make tests''`
	@echo ''` ''`
	@echo ''` To run some java tests type:''`
	@echo ''` ''`
	@echo ''`   make javatests''`
	@echo ''` ''`
	@echo ''` The tests assume that several data files are in the directories''` 
	@echo ''` $(DATADIRI) and $(DATADIRO)''`
	@echo ''` ''`
	@echo ''` Alternatively tests can be run comparing MD5 signatures only by''` 
	@echo ''` ''`
	@echo ''`   make tests_sigs_only''`
	@echo ''` ''`
	@echo ''` These signature only tests save space and download time by''`
	@echo ''` assuming that input data files and the output signatures''`
	@echo ''` are in the directories''` 
	@echo ''`  $(DATADIRI) and $(DATADIRS)''`
	@echo ''` ''`
	@echo ''` These directory can be obtained from''`
	@echo ''` ''`
	@echo ''`   $(DATAURLI) ''`
	@echo ''`   $(DATAURLO) ''`
	@echo ''`   $(DATAURLS) ''`
	@echo ''` ''`
	@echo ''` To clean up the directories type:''`
	@echo ''` ''`
	@echo ''`   make clean''`
	@echo ''` ''`
	@echo ''` To install the library and binaries type:''`
	@echo ''` ''`
	@echo ''`   make install''`
	@echo ''` ''`
	@echo ''`***************************************************************''`
	@echo ''` ''`

#
# Compile the library and examples
#
all::	$(BIN) $(SOURCE) $(F90SOURCE) $(HEADERS) \
'm4_ifelse(cbf_use_pycifrw,`yes',`
		$(PYCIFRW) $(PLY) \
')m4_dnl
`		symlinksdone $(REGEXDEP) \
		$(LIB)/libcbf.a          \
		$(LIB)/libfcb.a          \
		$(LIB)/libimg.a          \
		$(BIN)/adscimg2cbf       \
		$(BIN)/cbf2adscimg       \
		$(BIN)/convert_image     \
		$(BIN)/convert_minicbf   \
		$(BIN)/sequence_match    \
		$(BIN)/arvai_test        \
		$(BIN)/makecbf           \
		$(BIN)/img2cif           \
		$(BIN)/adscimg2cbf       \
		$(BIN)/cif2cbf           \
		$(BIN)/testcell          \
		$(BIN)/cif2c             \
		$(BIN)/testreals         \
		$(BIN)/testflat          \
		$(BIN)/testflatpacked


ifneq ($(F90C),)
all::	$(BIN)/test_xds_binary   \
		$(BIN)/test_fcb_read_image
endif

shared:	$(SOLIB)/libcbf.so $(SOLIB)/libfcb.so $(SOLIB)/libimg.so

javawrapper: shared $(JCBF) $(JCBF)/cbflib-$(VERSION).jar $(SOLIB)/libcbf_wrap.so

ifneq ($(CBFLIB_USE_PYCIFRW),)
PYCIFRWDEF = -Dcbf_use_pycifrw=yes
else
PYCIFRWDEF =
endif

Makefiles: Makefile			 \
		Makefile_LINUX           \
		Makefile_LINUX_64        \
		Makefile_LINUX_gcc42     \
		Makefile_LINUX_DMALLOC   \
		Makefile_LINUX_gcc42_DMALLOC \
		Makefile_OSX             \
		Makefile_OSX_gcc42       \
		Makefile_OSX_gcc42_DMALLOC   \
		Makefile_AIX             \
		Makefile_MINGW           \
		Makefile_IRIX_gcc


Makefile_LINUX: $(M4)/Makefile.m4
		-cp Makefile_LINUX Makefile_LINUX_old
		m4 -P $(PYCIFRWDEF) -Dcbf_system=LINUX $(M4)/Makefile.m4 > Makefile_LINUX 

Makefile_LINUX_DMALLOC: $(M4)/Makefile.m4
		-cp Makefile_LINUX Makefile_LINUX_old
		m4 -P $(PYCIFRWDEF) -Dcbf_system=LINUX_DMALLOC $(M4)/Makefile.m4 > Makefile_LINUX_DMALLOC

Makefile_LINUX_64: $(M4)/Makefile.m4
		-cp Makefile_LINUX_64 Makefile_LINUX_64_old
		m4 -P $(PYCIFRWDEF) -Dcbf_system=LINUX_64 $(M4)/Makefile.m4 > Makefile_LINUX_64

Makefile_LINUX_gcc42: $(M4)/Makefile.m4
		-cp Makefile_LINUX_gcc42 Makefile_LINUX_gcc42_old
		m4 -P $(PYCIFRWDEF) -Dcbf_system=LINUX_gcc42 $(M4)/Makefile.m4 > Makefile_LINUX_gcc42 

Makefile_LINUX_gcc42_DMALLOC: $(M4)/Makefile.m4
		-cp Makefile_LINUX_gcc42 Makefile_LINUX_gcc42_old
		m4 -P $(PYCIFRWDEF) -Dcbf_system=LINUX_gcc42_DMALLOC $(M4)/Makefile.m4 > Makefile_LINUX_gcc42_DMALLOC 

Makefile_OSX: $(M4)/Makefile.m4
		-cp Makefile_OSX Makefile_OSX_old
		m4 -P $(PYCIFRWDEF) -Dcbf_system=OSX $(M4)/Makefile.m4 > Makefile_OSX 

Makefile_OSX_gcc42: $(M4)/Makefile.m4
		-cp Makefile_OSX_gcc42 Makefile_OSX_gcc42_old
		m4 -P $(PYCIFRWDEF) -Dcbf_system=OSX_gcc42 $(M4)/Makefile.m4 > Makefile_OSX_gcc42 

Makefile_OSX_gcc42_DMALLOC: $(M4)/Makefile.m4
		-cp Makefile_OSX_gcc42 Makefile_OSX_gcc42_old
		m4 -P $(PYCIFRWDEF) -Dcbf_system=OSX_gcc42_DMALLOC $(M4)/Makefile.m4 > Makefile_OSX_gcc42_DMALLOC

Makefile_AIX: $(M4)/Makefile.m4
		-cp Makefile_AIX Makefile_AIX_old
		m4 -P $(PYCIFRWDEF) -Dcbf_system=AIX $(M4)/Makefile.m4 > Makefile_AIX 

Makefile_MINGW: $(M4)/Makefile.m4
		-cp Makefile_MINGW Makefile_MINGW_old
		m4 -P $(PYCIFRWDEF) -Dcbf_system=MINGW $(M4)/Makefile.m4 > Makefile_MINGW 

Makefile_IRIX_gcc: $(M4)/Makefile.m4
		-cp Makefile_IRIX_gcc Makefile_IRIX_gcc_old
		m4 -P $(PYCIFREDEF) -Dcbf_system=IRIX_gcc $(M4)/Makefile.m4 > Makefile_IRIX_gcc
		
Makefile: $(M4)/Makefile.m4
		-cp Makefile Makefile_old
		m4 -P $(PYCIFRWDEF) -Dcbf_system=default $(M4)/Makefile.m4 > Makefile 

symlinksdone:
	chmod a+x .symlinks
	chmod a+x .undosymlinks
	chmod a+x doc/.symlinks
	chmod a+x doc/.undosymlinks
	chmod a+x libtool/.symlinks
	chmod a+x libtool/.undosymlinks
	./.symlinks $(SLFLAGS)
	touch symlinksdone

install:  all $(INSTALLDIR) $(INSTALLDIR)/lib $(INSTALLDIR)/bin \
			$(INSTALLDIR)/include $(INSTALLDIR)/include/cbflib \
			$(PYSOURCE)
		-chmod -R 755 $(INSTALLDIR)/include/cbflib
		-chmod 755 $(INSTALLDIR)/lib/libcbf.a
		-cp $(INSTALLDIR)/lib/libcbf.a $(INSTALLDIR)/lib/libcbf_old.a
		cp $(LIB)/libcbf.a $(INSTALLDIR)/lib/libcbf.a
		-cp $(INSTALLDIR)/lib/libimg.a $(INSTALLDIR)/lib/libimg_old.a
		cp $(LIB)/libimg.a $(INSTALLDIR)/lib/libimg.a
		-cp $(INSTALLDIR)/bin/adscimg2cbf $(INSTALLDIR)/bin/adscimg2cbf_old
		cp $(BIN)/adscimg2cbf $(INSTALLDIR)/bin/adscimg2cbf
		-cp $(INSTALLDIR)/bin/cbf2adscimg $(INSTALLDIR)/bin/cbf2adscimg_old
		cp $(BIN)/cbf2adscimg $(INSTALLDIR)/bin/cbf2adscimg
		-cp $(INSTALLDIR)/bin/convert_image $(INSTALLDIR)/bin/convert_image_old
		cp $(BIN)/convert_image $(INSTALLDIR)/bin/convert_image
		-cp $(INSTALLDIR)/bin/convert_minicbf $(INSTALLDIR)/bin/convert_minicbf_old
		cp $(BIN)/convert_minicbf $(INSTALLDIR)/bin/convert_minicbf
		-cp $(INSTALLDIR)/bin/makecbf $(INSTALLDIR)/bin/makecbf_old
		cp $(BIN)/makecbf $(INSTALLDIR)/bin/makecbf
		-cp $(INSTALLDIR)/bin/img2cif $(INSTALLDIR)/bin/img2cif_old
		cp $(BIN)/img2cif $(INSTALLDIR)/bin/img2cif
		-cp $(INSTALLDIR)/bin/cif2cbf $(INSTALLDIR)/bin/cif2cbf_old
		cp $(BIN)/cif2cbf $(INSTALLDIR)/bin/cif2cbf
		-cp $(INSTALLDIR)/bin/sequence_match $(INSTALLDIR)/bin/sequence_match_old
		cp $(BIN)/sequence_match $(INSTALLDIR)/bin/sequence_match
		-cp $(INSTALLDIR)/bin/arvai_test $(INSTALLDIR)/bin/arvai_test_old
		cp $(BIN)/arvai_test $(INSTALLDIR)/bin/arvai_test
		-cp $(INSTALLDIR)/bin/cif2c $(INSTALLDIR)/bin/cif2c_old
		cp $(BIN)/cif2c $(INSTALLDIR)/bin/cif2c
		-cp $(INSTALLDIR)/bin/testreals $(INSTALLDIR)/bin/testreals_old
		cp $(BIN)/testreals $(INSTALLDIR)/bin/testreals
		-cp $(INSTALLDIR)/bin/testflat $(INSTALLDIR)/bin/testflat_old
		cp $(BIN)/testflat $(INSTALLDIR)/bin/testflat
		-cp $(INSTALLDIR)/bin/testflatpacked $(INSTALLDIR)/bin/testflatpacked_old
		cp $(BIN)/testflatpacked $(INSTALLDIR)/bin/testflatpacked
'm4_ifelse(cbf_use_pycifrw,`yes',`

		cp $(SRC)/drel_lex.py $(INSTALLDIR)/bin/drel_lex.py
		cp $(SRC)/drel_yacc.py $(INSTALLDIR)/bin/drel_yacc.py
		cp $(SRC)/drelc.py $(INSTALLDIR)/bin/drelc.py
		cp $(SRC)/drel_prep.py $(INSTALLDIR)/bin/drel_prep.py
')m4_dnl
`		chmod -R 755 $(INSTALLDIR)/include/cbflib
		-rm -rf $(INSTALLDIR)/include/cbflib_old
		-cp -r $(INSTALLDIR)/include/cbflib $(INSTALLDIR)/include/cbflib_old
		-rm -rf $(INSTALLDIR)/include/cbflib
		cp -r $(INCLUDE) $(INSTALLDIR)/include/cbflib
		chmod 644 $(INSTALLDIR)/lib/libcbf.a
		chmod 755 $(INSTALLDIR)/bin/convert_image
		chmod 755 $(INSTALLDIR)/bin/convert_minicbf
		chmod 755 $(INSTALLDIR)/bin/makecbf
		chmod 755 $(INSTALLDIR)/bin/img2cif
		chmod 755 $(INSTALLDIR)/bin/cif2cbf
		chmod 755 $(INSTALLDIR)/bin/sequence_match
		chmod 755 $(INSTALLDIR)/bin/arvai_test
		chmod 755 $(INSTALLDIR)/bin/cif2c
		chmod 755 $(INSTALLDIR)/bin/testreals
		chmod 755 $(INSTALLDIR)/bin/testflat
		chmod 755 $(INSTALLDIR)/bin/testflatpacked
		chmod 644 $(INSTALLDIR)/include/cbflib/*.h
		
'm4_ifelse(cbf_use_pycifrw,`yes',`
#
# PyCifRW
#
$(PYCIFRW):
	$(DOWNLOAD) $(PYCIFRWURL)
	tar -xvf $(PYCIFRW).tar.gz
	-rm $(PYCIFRW).tar.gz
	(cd $(PYCIFRW); python setup.py install )

#
# PLY
#
$(PLY):
	$(DOWNLOAD) $(PLYURL)
	tar -xvf $(PLY).tar.gz
	-rm $(PLY).tar.gz
	(cd $(PLY); python setup.py install )
')m4_dnl
`
#
# REGEX
#

ifneq ($(REGEXDEP),)
$(REGEXDEP):	$(REGEX)
	(cd $(REGEX); ./configure; make install)
endif


$(REGEX):
	$(DOWNLOAD) $(REGEXURL)
	tar -xvf $(REGEX).tar.gz
	-rm $(REGEX).tar.gz
	

#
# TIFF
#
$(TIFF):
	$(DOWNLOAD) $(TIFFURL)
	tar -xvf $(TIFF).tar.gz
	-rm $(TIFF).tar.gz
	(cd $(TIFF); ./configure --prefix=$(TIFFPREFIX); make install)
	

#
# Directories
#
$(INSTALLDIR):
	mkdir -p $(INSTALLDIR)

$(INSTALLDIR)/lib:  $(INSTALLDIR)
	mkdir -p $(INSTALLDIR)/lib

$(INSTALLDIR)/bin:  $(INSTALLDIR)
	mkdir -p $(INSTALLDIR)/bin

$(INSTALLDIR)/include:  $(INSTALLDIR)
	mkdir -p $(INSTALLDIR)/include
	
$(INSTALLDIR)/include/cbflib: $(INSTALLDIR)/include
	mkdir -p $(INSTALLDIR)/include/cbflib



$(LIB):
	mkdir $@

$(BIN):
	mkdir $@

$(SOLIB):
	mkdir $@

$(JCBF):
	mkdir $@

#
# Parser
#
$(SRC)/cbf_stx.c: $(SRC)/cbf.stx.y
	bison $(SRC)/cbf.stx.y -o $(SRC)/cbf.stx.tab.c -d
	mv $(SRC)/cbf.stx.tab.c $(SRC)/cbf_stx.c
	mv $(SRC)/cbf.stx.tab.h $(INCLUDE)/cbf_stx.h

#
# CBF library
#
$(LIB)/libcbf.a: $(SOURCE) $(HEADERS) $(COMMONDEP) $(LIB)
	$(CC) $(CFLAGS) $(NOLLFLAG) $(PYCIFRWFLAG) $(INCLUDES) $(WARNINGS) -c $(SOURCE)
	$(AR) cr $@ *.o
	mv *.o $(LIB)
ifneq ($(RANLIB),)
	$(RANLIB) $@
endif

$(SOLIB)/libcbf.so: $(SOURCE) $(HEADERS) $(COMMONDEP) $(SOLIB)
	$(CC) $(CFLAGS) $(NOLLFLAG) $(PYCIFRWFLAG) $(SOCFLAGS) $(INCLUDES) $(WARNINGS) -c $(SOURCE)
	$(CC) -o $@ *.o $(SOLDFLAGS) $(EXTRALIBS)
	rm *.o

#
# IMG library
#
$(LIB)/libimg.a: $(EXAMPLES)/img.c $(HEADERS) $(COMMONDEP) $(LIB)
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) -c $(EXAMPLES)/img.c
	$(AR) cr $@ img.o
ifneq ($(RANLIB),)
	$(RANLIB) $@
endif
	rm img.o
	
$(SOLIB)/libimg.so: $(SOURCE) $(HEADERS) $(COMMONDEP) $(SOLIB)
	$(CC) $(CFLAGS) $(NOLLFLAG) $(SOCFLAGS) $(INCLUDES) $(WARNINGS) -c $(EXAMPLES)/img.c
	$(CC) -o $@ img.o $(SOLDFLAGS)
	rm img.o

#
# CBF and IMG libraries
#
CBF_IMG_LIBS:  $(LIB)/libcbf.a $(LIB)/libimg.a 	


#
# FCB library
#
$(LIB)/libfcb.a: $(F90SOURCE) $(COMMONDEP) $(LIB)
ifneq ($(F90C),)
	$(F90C) $(F90FLAGS) -c $(F90SOURCE)
	$(AR) cr $@ *.o
ifneq ($(RANLIB),)
	$(RANLIB) $@
endif
	rm *.o
else
	echo "Define F90C to build $(LIB)/libfcb.a"
endif

$(SOLIB)/libfcb.so: $(F90SOURCE) $(HEADERS) $(COMMONDEP) $(SOLIB)
ifneq ($(F90C),)
	$(F90C) $(F90FLAGS) $(SOCFLAGS) $(INCLUDES) $(WARNINGS) -c $(F90SOURCE)
	$(F90C) $(F90FLAGS) -o $@ *.o $(SOLDFLAGS)
	rm *.o
else
	echo "Define F90C to build $(SOLIB)/libfcb.so"
endif


#
# Python bindings
#
$(PYCBF)/_pycbf.$(PYCBFEXT): $(PYCBF)  $(LIB)/libcbf.a \
                              $(PYCBF)/$(SETUP_PY)     \
			                  $(LIB)/libfcb.a          \
			                  $(LIB)/libimg.a          \
			                  $(PYCBF)/pycbf.i         \
			                  $(PYCBF)/cbfhandlewrappers.i   \
			                  $(PYCBF)/cbfdetectorwrappers.i \
			                  $(PYCBF)/cbfgenericwrappers.i  \
			                  $(PYCBF)/cbfgoniometerwrappers.i
	(cd $(PYCBF); python $(SETUP_PY) build $(PYCBFBOPT); cp build/lib.*/_pycbf.$(PYCBFEXT) .) 

$(PYCBF)/setup.py: $(M4)/setup_py.m4
	(m4 -P -Dregexlib=NOREGEXLIB -Dregexlibdir=NOREGEXLIBDIR $(M4)/setup_py.m4 > $@)

$(PYCBF)/setup_MINGW.py: m4/setup_py.m4
	(m4 -P -Dregexlib=regex -Dregexlibdir=$(REGEXDIR) $(M4)/setup_py.m4 > $@)

$(LIB)/_pycbf.$(PYCBFEXT): $(PYCBF)/_pycbf.$(PYCBFEXT)
	cp $(PYCBF)/_pycbf.$(PYCBFEXT) $(LIB)/_pycbf.$(PYCBFEXT)
	
$(PYCBF)/pycbf.pdf: $(PYCBF)/pycbf.w
	(cd $(PYCBF); \
	$(NUWEB) pycbf; \
	latex pycbf; \
	$(NUWEB) pycbf; \
	latex pycbf; \
	dvipdfm pycbf )
	
$(PYCBF)/CBFlib.txt: $(DOC)/CBFlib.html
	links -dump $(DOC)/CBFlib.html > $(PYCBF)/CBFlib.txt

$(PYCBF)/cbfhandlewrappers.i   \
$(PYCBF)/cbfdetectorwrappers.i \
$(PYCBF)/cbfgenericwrappers.i  \
$(PYCBF)/cbfgoniometerwrappers.i:  $(PYCBF)/CBFlib.txt $(PYCBF)/make_pycbf.py
	(cd $(PYCBF);  python make_pycbf.py; $(PYSWIG) pycbf.i; python setup.py build)


#
# Java bindings
#
$(JCBF)/cbflib-$(VERSION).jar: $(JCBF) $(JCBF)/jcbf.i
	$(JSWIG) -I$(INCLUDE) -package org.iucr.cbflib -outdir $(JCBF) $(JCBF)/jcbf.i
	$(JAVAC) -d . $(JCBF)/*.java
	$(JAR) cf $@ org

$(SOLIB)/libcbf_wrap.so: $(JCBF)/cbflib-$(VERSION).jar $(SOLIB)/libcbf.so
	$(CC) $(CFLAGS) $(NOLLFLAG) $(SOCFLAGS) $(INCLUDES) $(WARNINGS) $(JAVAINCLUDES) -c $(JCBF)/jcbf_wrap.c
	$(CC) -o $@ jcbf_wrap.o $(SOLDFLAGS) -L$(SOLIB) -lcbf
	rm jcbf_wrap.o

#
# F90SOURCE
#
$(SRC)/fcb_exit_binary.f90: $(M4)/fcb_exit_binary.m4 $(M4)/fcblib_defines.m4
	(cd $(M4); m4 -P $(M4FLAGS) fcb_exit_binary.m4) > $(SRC)/fcb_exit_binary.f90
$(SRC)/fcb_next_binary.f90: $(M4)/fcb_next_binary.m4 $(M4)/fcblib_defines.m4
	(cd $(M4); m4 -P $(M4FLAGS) fcb_next_binary.m4) > $(SRC)/fcb_next_binary.f90
$(SRC)/fcb_open_cifin.f90: $(M4)/fcb_open_cifin.m4 $(M4)/fcblib_defines.m4
	(cd $(M4); m4 -P $(M4FLAGS) fcb_open_cifin.m4)  > $(SRC)/fcb_open_cifin.f90
$(SRC)/fcb_packed.f90: $(M4)/fcb_packed.m4 $(M4)/fcblib_defines.m4
	(cd $(M4); m4 -P $(M4FLAGS) fcb_packed.m4)      > $(SRC)/fcb_packed.f90
$(SRC)/fcb_read_bits.f90: $(M4)/fcb_read_bits.m4 $(M4)/fcblib_defines.m4
	(cd $(M4); m4 -P $(M4FLAGS) fcb_read_bits.m4)   > $(SRC)/fcb_read_bits.f90
$(SRC)/fcb_read_image.f90: $(M4)/fcb_read_image.m4 $(M4)/fcblib_defines.m4
	(cd $(M4); m4 -P $(M4FLAGS) fcb_read_image.m4)  > $(SRC)/fcb_read_image.f90
$(SRC)/fcb_read_xds_i2.f90: $(M4)/fcb_read_xds_i2.m4 $(M4)/fcblib_defines.m4
	(cd $(M4); m4 -P $(M4FLAGS) fcb_read_xds_i2.m4) > $(SRC)/fcb_read_xds_i2.f90
$(EXAMPLES)/test_fcb_read_image.f90: $(M4)/test_fcb_read_image.m4 $(M4)/fcblib_defines.m4
	(cd $(M4); m4 -P $(M4FLAGS) test_fcb_read_image.m4) > $(EXAMPLES)/test_fcb_read_image.f90
$(EXAMPLES)/test_xds_binary.f90: $(M4)/test_xds_binary.m4 $(M4)/fcblib_defines.m4
	(cd $(M4); m4 -P $(M4FLAGS) test_xds_binary.m4) > $(EXAMPLES)/test_xds_binary.f90

#
# convert_image example program
#
$(BIN)/convert_image: $(LIB)/libcbf.a $(EXAMPLES)/convert_image.c $(EXAMPLES)/img.c \
					$(GOPTLIB)	$(GOPTINC)
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/convert_image.c $(EXAMPLES)/img.c $(GOPTLIB) -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@
#
# convert_minicbf example program
#
$(BIN)/convert_minicbf: $(LIB)/libcbf.a $(EXAMPLES)/convert_minicbf.c \
					$(GOPTLIB)	$(GOPTINC)
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/convert_minicbf.c $(GOPTLIB) -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@

#
# makecbf example program
#
$(BIN)/makecbf: $(LIB)/libcbf.a $(EXAMPLES)/makecbf.c $(LIB)/libimg.a
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/makecbf.c  -L$(LIB) \
		  -lcbf $(EXTRALIBS) -limg -o $@


#
# adscimg2cbf example program
#
$(BIN)/adscimg2cbf: $(LIB)/libcbf.a $(EXAMPLES)/adscimg2cbf.c $(EXAMPLES)/adscimg2cbf_sub.c
	$(CC) $(CFLAGS) $(NOLLFLAG) -D_SVID_SOURCE $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/adscimg2cbf.c $(EXAMPLES)/adscimg2cbf_sub.c  -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@

#
# cbf2adscimg example program
#
$(BIN)/cbf2adscimg: $(LIB)/libcbf.a $(EXAMPLES)/cbf2adscimg.c $(EXAMPLES)/cbf2adscimg_sub.c
	$(CC) $(CFLAGS) $(NOLLFLAG) -D_SVID_SOURCE $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/cbf2adscimg.c $(EXAMPLES)/cbf2adscimg_sub.c  -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@

#
# changtestcompression example program
#
$(BIN)/changtestcompression: $(LIB)/libcbf.a $(EXAMPLES)/changtestcompression.c
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/changtestcompression.c -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@

#
# img2cif example program
#
$(BIN)/img2cif: $(LIB)/libcbf.a $(EXAMPLES)/img2cif.c $(LIB)/libimg.a \
					$(GOPTLIB) 	$(GOTPINC)
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/img2cif.c $(GOPTLIB) -L$(LIB) \
		  -lcbf $(EXTRALIBS) -limg -o $@

#
# cif2cbf example program
#
$(BIN)/cif2cbf: $(LIB)/libcbf.a $(EXAMPLES)/cif2cbf.c $(LIB)/libimg.a \
					$(GOPTLIB)	$(GOPTINC)
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/cif2cbf.c $(GOPTLIB) -L$(LIB) \
		  -lcbf $(EXTRALIBS) -limg -o $@
#
# dectris cbf_template_t program
#
$(BIN)/cbf_template_t: $(DECTRIS_EXAMPLES)/cbf_template_t.c \
			$(DECTRIS_EXAMPLES)/mx_cbf_t_extras.h \
			$(DECTRIS_EXAMPLES)/mx_parms.h
	$(CC) $(CFLAGS) $(NOLLFLAG) -I $(DECTRIS_EXAMPLES)  $(WARNINGS) \
			$(DECTRIS_EXAMPLES)/cbf_template_t.c -o $@

#
# testcell example program
#
$(BIN)/testcell: $(LIB)/libcbf.a $(EXAMPLES)/testcell.C
	$(C++) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/testcell.C -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@

#
# cif2c example program
#
$(BIN)/cif2c: $(LIB)/libcbf.a $(EXAMPLES)/cif2c.c
	$(C++) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/cif2c.c -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@

#
# sauter_test example program
#
$(BIN)/sauter_test: $(LIB)/libcbf.a $(EXAMPLES)/sauter_test.C
	$(C++) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/sauter_test.C -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@

#
# sequence_match example program
#
$(BIN)/sequence_match: $(LIB)/libcbf.a $(EXAMPLES)/sequence_match.c $(LIB)/libimg.a \
					$(GOPTLIB)	$(GOPTINC)
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/sequence_match.c $(GOPTLIB) -L$(LIB) \
		  -lcbf $(EXTRALIBS) -limg -o $@

#
# tiff2cbf example program
#
$(BIN)/tiff2cbf: $(LIB)/libcbf.a $(EXAMPLES)/tiff2cbf.c \
					$(GOPTLIB)	$(GOPTINC) $(TIFF)
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
		  -I$(TIFFPREFIX)/include $(EXAMPLES)/tiff2cbf.c $(GOPTLIB) -L$(LIB) \
		    -lcbf -L$(TIFFPREFIX)/lib -ltiff $(EXTRALIBS) -limg -o $@

#
# Andy Arvai''`s buffered read test program
#
$(BIN)/arvai_test: $(LIB)/libcbf.a $(EXAMPLES)/arvai_test.c $(LIB)/libimg.a \
					$(GOPTLIB)	$(GOPTINC)
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/arvai_test.c $(GOPTLIB) -L$(LIB) \
		  -lcbf $(EXTRALIBS) -limg -o $@

#
# testreals example program
#
$(BIN)/testreals: $(LIB)/libcbf.a $(EXAMPLES)/testreals.c
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/testreals.c -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@

#
# testflat example program
#
$(BIN)/testflat: $(LIB)/libcbf.a $(EXAMPLES)/testflat.c
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/testflat.c -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@
#
# testflatpacked example program
#
$(BIN)/testflatpacked: $(LIB)/libcbf.a $(EXAMPLES)/testflatpacked.c
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/testflatpacked.c -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@

ifneq ($(F90C),)
#
# test_xds_binary example program
#
$(BIN)/test_xds_binary: $(LIB)/libfcb.a $(EXAMPLES)/test_xds_binary.f90
	$(F90C) $(F90FLAGS) $(F90LDFLAGS) $(EXAMPLES)/test_xds_binary.f90 \
		-L$(LIB) -lfcb -o $@

#
# test_fcb_read_image example program
#
$(BIN)/test_fcb_read_image: $(LIB)/libfcb.a $(EXAMPLES)/test_fcb_read_image.f90
	$(F90C) $(F90FLAGS) $(F90LDFLAGS) $(EXAMPLES)/test_fcb_read_image.f90 \
		-L$(LIB) -lfcb -o $@
endif

#
# testcbf (C)
#
$(BIN)/ctestcbf: $(EXAMPLES)/testcbf.c $(LIB)/libcbf.a
	$(CC) $(CFLAGS) $(NOLLFLAG) $(INCLUDES) $(WARNINGS) \
			  $(EXAMPLES)/testcbf.c -L$(LIB) \
		  -lcbf $(EXTRALIBS) -o $@

#
# testcbf (Java)
#
$(BIN)/testcbf.class: $(EXAMPLES)/testcbf.java $(JCBF)/cbflib-$(VERSION).jar $(SOLIB)/libcbf_wrap.so
	$(JAVAC) -cp $(JCBF)/cbflib-$(VERSION).jar -d $(BIN) $(EXAMPLES)/testcbf.java


#
# Data files for tests
#

$(DATADIRI):
		(cd ..; $(DOWNLOAD) $(DATAURLI))
		(cd ..; tar -zxvf CBFlib_$(VERSION)_Data_Files_Input.tar.gz)
		-(cd ..; rm CBFlib_$(VERSION)_Data_Files_Input.tar.gz)

$(DATADIRO):
		(cd ..; $(DOWNLOAD) $(DATAURLO))
		(cd ..; tar -zxvf CBFlib_$(VERSION)_Data_Files_Output.tar.gz)
		-(cd ..; rm CBFlib_$(VERSION)_Data_Files_Output.tar.gz)

$(DATADIRS):
		(cd ..; $(DOWNLOAD) $(DATAURLS))
		(cd ..; tar -zxvf CBFlib_$(VERSION)_Data_Files_Output_Sigs_Only.tar.gz)
		-(cd ..; rm CBFlib_$(VERSION)_Data_Files_Output_Sigs_Only.tar.gz)


# Input Data Files 

TESTINPUT_BASIC =  example.mar2300
DATADIRI_INPUT_BASIC = $(DATADIRI)/example.mar2300$(CEXT)


TESTINPUT_EXTRA =  9ins.cif mb_LP_1_001.img insulin_pilatus6m.cbf testrealin.cbf \
		testflatin.cbf testflatpackedin.cbf XRD1621.tif
DATADIRI_INPUT_EXTRA = $(DATADIRI)/9ins.cif$(CEXT) $(DATADIRI)/mb_LP_1_001.img$(CEXT) \
		$(DATADIRI)/insulin_pilatus6m.cbf$(CEXT) $(DATADIRI)/testrealin.cbf$(CEXT) \
		$(DATADIRI)/testflatin.cbf$(CEXT) $(DATADIRI)/testflatpackedin.cbf$(CEXT) \
		$(DATADIRI)/XRD1621.tif$(CEXT)


# Output Data Files

TESTOUTPUT =  adscconverted_flat_orig.cbf \
		adscconverted_orig.cbf converted_flat_orig.cbf converted_orig.cbf \
		insulin_pilatus6mconverted_orig.cbf \
		mb_LP_1_001_orig.cbf testcell_orig.prt \
		test_xds_bin_testflatout_orig.out \
		test_xds_bin_testflatpackedout_orig.out test_fcb_read_testflatout_orig.out \
		test_fcb_read_testflatpackedout_orig.out \
		XRD1621_orig.cbf XRD1621_I4encbC100_orig.cbf
NEWTESTOUTPUT = adscconverted_flat.cbf \
		adscconverted.cbf converted_flat.cbf converted.cbf \
		insulin_pilatus6mconverted.cbf \
		mb_LP_1_001.cbf testcell.prt \
		test_xds_bin_testflatout.out \
		test_xds_bin_testflatpackedout.out test_fcb_read_testflatout.out \
		test_fcb_read_testflatpackedout.out \
		XRD1621.cbf XRD1621_I4encbC100.cbf
DATADIRO_OUTPUT =  $(DATADIRO)/adscconverted_flat_orig.cbf$(CEXT) \
		$(DATADIRO)/adscconverted_orig.cbf$(CEXT) \
		$(DATADIRO)/converted_flat_orig.cbf$(CEXT) \
		$(DATADIRO)/converted_orig.cbf$(CEXT) \
		$(DATADIRO)/insulin_pilatus6mconverted_orig.cbf$(CEXT) \
		$(DATADIRO)/mb_LP_1_001_orig.cbf$(CEXT) \
		$(DATADIRO)/testcell_orig.prt$(CEXT) \
		$(DATADIRO)/test_xds_bin_testflatout_orig.out$(CEXT) \
		$(DATADIRO)/test_xds_bin_testflatpackedout_orig.out$(CEXT) \
		$(DATADIRO)/test_fcb_read_testflatout_orig.out$(CEXT) \
		$(DATADIRO)/test_fcb_read_testflatpackedout_orig.out$(CEXT) \
		$(DATADIRO)/XRD1621_orig.cbf$(CEXT) \
		$(DATADIRO)/XRD1621_I4encbC100_orig.cbf$(CEXT)
DATADIRO_OUTPUT_SIGNATURES =  $(DATADIRO)/adscconverted_flat_orig.cbf$(SEXT) \
		$(DATADIRO)/adscconverted_orig.cbf$(SEXT) \
		$(DATADIRO)/converted_flat_orig.cbf$(SEXT) \
		$(DATADIRO)/converted_orig.cbf$(SEXT) \
		$(DATADIRO)/insulin_pilatus6mconverted_orig.cbf$(SEXT) \
		$(DATADIRO)/mb_LP_1_001_orig.cbf$(SEXT) \
		$(DATADIRO)/testcell_orig.prt$(SEXT) \
		$(DATADIRO)/test_xds_bin_testflatout_orig.out$(SEXT) \
		$(DATADIRO)/test_xds_bin_testflatpackedout_orig.out$(SEXT) \
		$(DATADIRO)/test_fcb_read_testflatout_orig.out$(SEXT) \
		$(DATADIRO)/test_fcb_read_testflatpackedout_orig.out$(SEXT) \
		$(DATADIRO)/XRD1621_orig.cbf$(SEXT) \
		$(DATADIRO)/XRD1621_I4encbC100_orig.cbf$(SEXT)


		
# Output Data File Signatures

TESTOUTPUTSIGS = adscconverted_flat_orig.cbf$(SEXT) \
		adscconverted_orig.cbf$(SEXT) converted_flat_orig.cbf$(SEXT) converted_orig.cbf$(SEXT) \
		insulin_pilatus6mconverted_orig.cbf$(SEXT) \
		mb_LP_1_001_orig.cbf$(SEXT) testcell_orig.prt$(SEXT) \
		test_xds_bin_testflatout_orig.out$(SEXT) \
		test_xds_bin_testflatpackedout_orig.out$(SEXT) test_fcb_read_testflatout_orig.out$(SEXT) \
		test_fcb_read_testflatpackedout_orig.out$(SEXT) \
		XRD1621_orig.cbf$(SEXT)
DATADIRS_OUTPUT_SIGNATURES =  $(DATADIRS)/adscconverted_flat_orig.cbf$(SEXT) \
		$(DATADIRS)/adscconverted_orig.cbf$(SEXT) \
		$(DATADIRS)/converted_flat_orig.cbf$(SEXT) \
		$(DATADIRS)/converted_orig.cbf$(SEXT) \
		$(DATADIRS)/insulin_pilatus6mconverted_orig.cbf$(SEXT) \
		$(DATADIRS)/mb_LP_1_001_orig.cbf$(SEXT) \
		$(DATADIRS)/testcell_orig.prt$(SEXT) \
		$(DATADIRS)/test_xds_bin_testflatout_orig.out$(SEXT) \
		$(DATADIRS)/test_xds_bin_testflatpackedout_orig.out$(SEXT) \
		$(DATADIRS)/test_fcb_read_testflatout_orig.out$(SEXT) \
		$(DATADIRS)/test_fcb_read_testflatpackedout_orig.out$(SEXT) \
		$(DATADIRS)/XRD1621_orig.cbf$(SEXT) \
		$(DATADIRS)/XRD1621_I4encbC100_orig.cbf$(SEXT)

# Fetch Input Data Files 

$(TESTINPUT_BASIC):	$(DATADIRI) $(DATADIRI_INPUT_BASIC)
		$(DECOMPRESS) < $(DATADIRI)/$@$(CEXT) > $@
		cp $(DATADIRI)/$@$(SEXT)  $@$(SEXT)
		-$(SIGNATURE) < $@ | $(DIFF) - $@$(SEXT)

$(TESTINPUT_EXTRA):	$(DATADIRI) $(DATADIRI_INPUT_EXTRA)
		$(DECOMPRESS) < $(DATADIRI)/$@$(CEXT) > $@
		cp $(DATADIRI)/$@$(SEXT)  $@$(SEXT)
		-$(SIGNATURE) < $@ | $(DIFF) - $@$(SEXT)


# Fetch Output Data Files and Signatures

$(TESTOUTPUT):	$(DATADIRO) $(DATADIRO_OUTPUT) $(DATADIRO_OUTPUT_SIGNATURES)
		$(DECOMPRESS) < $(DATADIRO)/$@$(CEXT) > $@
		cp $(DATADIRO)/$@$(SEXT) $@$(SEXT)
		-$(SIGNATURE) < $@ | $(DIFF) - $@$(SEXT)

# Fetch Output Data File Signatures

$(TESTOUTPUTSIGS):	$(DATADIRS) $(DATADIRS_OUTPUT_SIGNATURES)
		cp $(DATADIRS)/$@ $@



#
# Tests
#
tests:				$(LIB) $(BIN) symlinksdone basic extra dectristests pycbftests
tests_sigs_only:	$(LIB) $(BIN) symlinksdone basic extra_sigs_only
restore_output:		$(NEWTESTOUTPUT) $(DATADIRO)
		$(SIGNATURE) < adscconverted_flat.cbf > $(DATADIRO)/adscconverted_flat_orig.cbf$(SEXT)
		$(SIGNATURE) < adscconverted.cbf > $(DATADIRO)/adscconverted_orig.cbf$(SEXT)
		$(SIGNATURE) < converted_flat.cbf > $(DATADIRO)/converted_flat_orig.cbf$(SEXT)
		$(SIGNATURE) < converted.cbf > $(DATADIRO)/converted_orig.cbf$(SEXT)
		$(SIGNATURE) < insulin_pilatus6mconverted.cbf > $(DATADIRO)/insulin_pilatus6mconverted_orig.cbf$(SEXT)
		$(SIGNATURE) < mb_LP_1_001.cbf$ > $(DATADIRO)/mb_LP_1_001_orig.cbf$(SEXT)
		$(SIGNATURE) < testcell.prt > $(DATADIRO)/testcell_orig.prt$(SEXT)
		$(SIGNATURE) < test_xds_bin_testflatout.out > $(DATADIRO)/test_xds_bin_testflatout_orig.out$(SEXT)
		$(SIGNATURE) < test_xds_bin_testflatpackedout.out > $(DATADIRO)/test_xds_bin_testflatpackedout_orig.out$(SEXT)
		$(SIGNATURE) < test_fcb_read_testflatout.out > $(DATADIRO)/test_fcb_read_testflatout_orig.out$(SEXT)
		$(SIGNATURE) < test_fcb_read_testflatpackedout.out > $(DATADIRO)/test_fcb_read_testflatpackedout_orig.out$(SEXT)
		$(SIGNATURE) < XRD1621.cbf > $(DATADIRO)/XRD1621_orig.cbf$(SEXT)
		$(SIGNATURE) < XRD1621_I4encbC100.cbf > $(DATADIRO)/XRD1621_I4encbC100_orig.cbf$(SEXT)
		$(COMPRESS)  < adscconverted_flat.cbf > $(DATADIRO)/adscconverted_flat_orig.cbf$(CEXT)
		$(COMPRESS)  < adscconverted.cbf > $(DATADIRO)/adscconverted_orig.cbf$(CEXT)
		$(COMPRESS)  < converted_flat.cbf > $(DATADIRO)/converted_flat_orig.cbf$(CEXT)
		$(COMPRESS)  < converted.cbf > $(DATADIRO)/converted_orig.cbf$(CEXT)
		$(COMPRESS)  < insulin_pilatus6mconverted.cbf > $(DATADIRO)/insulin_pilatus6mconverted_orig.cbf$(CEXT)
		$(COMPRESS)  < mb_LP_1_001.cbf$ > $(DATADIRO)/mb_LP_1_001_orig.cbf$(CEXT)
		$(COMPRESS)  < testcell.prt > $(DATADIRO)/testcell_orig.prt$(CEXT)
		$(COMPRESS)  < test_xds_bin_testflatout.out > $(DATADIRO)/test_xds_bin_testflatout_orig.out$(CEXT)
		$(COMPRESS)  < test_xds_bin_testflatpackedout.out > $(DATADIRO)/test_xds_bin_testflatpackedout_orig.out$(CEXT)
		$(COMPRESS)  < test_fcb_read_testflatout.out > $(DATADIRO)/test_fcb_read_testflatout_orig.out$(CEXT)
		$(COMPRESS)  < test_fcb_read_testflatpackedout.out > $(DATADIRO)/test_fcb_read_testflatpackedout_orig.out$(CEXT)
		$(COMPRESS)  < XRD1621.cbf > $(DATADIRO)/XRD1621_orig.cbf$(CEXT)
		$(COMPRESS)  < XRD1621_I4encbC100.cbf > $(DATADIRO)/XRD1621_I4encbC100_orig.cbf$(CEXT)

restore_sigs_only:	$(NEWTESTOUTPUT) $(DATADIRS)
		$(SIGNATURE) < adscconverted_flat.cbf > $(DATADIRS)/adscconverted_flat_orig.cbf$(SEXT)
		$(SIGNATURE) < adscconverted.cbf > $(DATADIRS)/adscconverted_orig.cbf$(SEXT)
		$(SIGNATURE) < converted_flat.cbf > $(DATADIRS)/converted_flat_orig.cbf$(SEXT)
		$(SIGNATURE) < converted.cbf > $(DATADIRS)/converted_orig.cbf$(SEXT)
		$(SIGNATURE) < insulin_pilatus6mconverted.cbf > $(DATADIRS)/insulin_pilatus6mconverted_orig.cbf$(SEXT)
		$(SIGNATURE) < mb_LP_1_001.cbf$ > $(DATADIRS)/mb_LP_1_001_orig.cbf$(SEXT)
		$(SIGNATURE) < testcell.prt > $(DATADIRS)/testcell_orig.prt$(SEXT)
		$(SIGNATURE) < test_xds_bin_testflatout.out > $(DATADIRS)/test_xds_bin_testflatout_orig.out$(SEXT)
		$(SIGNATURE) < test_xds_bin_testflatpackedout.out > $(DATADIRS)/test_xds_bin_testflatpackedout_orig.out$(SEXT)
		$(SIGNATURE) < test_fcb_read_testflatout.out > $(DATADIRS)/test_fcb_read_testflatout_orig.out$(SEXT)
		$(SIGNATURE) < test_fcb_read_testflatpackedout.out > $(DATADIRS)/test_fcb_read_testflatpackedout_orig.out$(SEXT)
		$(SIGNATURE) < XRD1621.cbf > $(DATADIRS)/XRD1621_orig.cbf$(SEXT)
		$(SIGNATURE) < XRD1621_I4encbC100.cbf > $(DATADIRS)/XRD1621_I4encbC100_orig.cbf$(SEXT)
restore_signatures:	restore_output restore_sigs_only
		
#
# Basic Tests
#

basic:	$(BIN)/makecbf $(BIN)/img2cif $(BIN)/cif2cbf $(TESTINPUT_BASIC)
	$(BIN)/makecbf example.mar2300 makecbf.cbf
	$(BIN)/img2cif -c flatpacked -m headers -d digest \
		-e base64  example.mar2300  img2cif_packed.cif
	$(BIN)/img2cif -c canonical -m headers -d digest \
		-e base64  example.mar2300  img2cif_canonical.cif
	$(BIN)/img2cif -c flatpacked -m headers -d digest \
		-e none  example.mar2300  img2cif_packed.cbf
	$(BIN)/img2cif -c canonical -m headers -d digest \
		-e none  example.mar2300  img2cif_canonical.cbf
	$(BIN)/cif2cbf -e none -c flatpacked \
		img2cif_canonical.cif cif2cbf_packed.cbf
	$(BIN)/cif2cbf -e none -c canonical \
		img2cif_packed.cif cif2cbf_canonical.cbf
	-cmp cif2cbf_packed.cbf    makecbf.cbf
	-cmp cif2cbf_packed.cbf    img2cif_packed.cbf
	-cmp cif2cbf_canonical.cbf img2cif_canonical.cbf


#
# Extra Tests
#
ifneq ($(F90C),)
extra:	$(BIN)/convert_image $(BIN)/convert_minicbf $(BIN)/cif2cbf $(BIN)/testcell \
	$(BIN)/testreals $(BIN)/testflat $(BIN)/testflatpacked \
	$(BIN)/test_xds_binary $(BIN)/test_fcb_read_image $(BIN)/convert_minicbf \
	$(BIN)/sauter_test $(BIN)/adscimg2cbf $(BIN)/cbf2adscimg \
	$(BIN)/changtestcompression $(BIN)/tiff2cbf \
	basic $(TESTINPUT_EXTRA) $(TESTOUTPUT)
else
extra:	$(BIN)/convert_image $(BIN)/convert_minicbf $(BIN)/cif2cbf $(BIN)/testcell \
	$(BIN)/testreals $(BIN)/testflat $(BIN)/testflatpacked \
	$(BIN)/convert_minicbf \
	$(BIN)/sauter_test $(BIN)/adscimg2cbf $(BIN)/cbf2adscimg \
	basic $(TESTINPUT_EXTRA) $(TESTOUTPUT)
endif
	$(TIME) $(BIN)/cif2cbf -e hex -c none \
		makecbf.cbf cif2cbf_ehcn.cif
	$(TIME) $(BIN)/cif2cbf -e none -c flatpacked \
		cif2cbf_ehcn.cif cif2cbf_encp.cbf; rm cif2cbf_ehcn.cif
	-cmp makecbf.cbf cif2cbf_encp.cbf
	$(TIME) $(BIN)/cif2cbf -i 9ins.cif -o 9ins.cbf
	-cmp 9ins.cif 9ins.cbf
	$(TIME) $(BIN)/convert_image -F example.mar2300 converted_flat.cbf
	-cmp converted_flat.cbf converted_flat_orig.cbf
	$(TIME) $(BIN)/convert_image example.mar2300 converted.cbf
	-cmp converted.cbf converted_orig.cbf
	-$(TIME) $(BIN)/testcell < testcell.dat > testcell.prt
	-cmp testcell.prt testcell_orig.prt
	$(TIME) $(BIN)/convert_image -F -d adscquantum315 mb_LP_1_001.img adscconverted_flat.cbf
	-cmp adscconverted_flat.cbf adscconverted_flat_orig.cbf
	$(TIME) $(BIN)/convert_image -d adscquantum315 mb_LP_1_001.img adscconverted.cbf
	-cmp adscconverted.cbf adscconverted_orig.cbf
	$(TIME) $(BIN)/adscimg2cbf --no_pad  --cbf_packed,flat mb_LP_1_001.img
	-cmp mb_LP_1_001.cbf mb_LP_1_001_orig.cbf
ifneq ($(CLEANTESTS),)
	mv mb_LP_1_001.cbf nmb_LP_1_001.cbf
else
	cp mb_LP_1_001.cbf nmb_LP_1_001.cbf
endif
	$(TIME) $(BIN)/cbf2adscimg nmb_LP_1_001.cbf
	-cmp nmb_LP_1_001.img mb_LP_1_001.img
	rm nmb_LP_1_001.cbf
ifneq ($(CLEANTESTS),)
	rm nmb_LP_1_001.img
endif
	$(TIME) $(BIN)/convert_minicbf -d pilatus6m insulin_pilatus6m.cbf insulin_pilatus6mconverted.cbf
	-cmp insulin_pilatus6mconverted.cbf insulin_pilatus6mconverted_orig.cbf
	$(TIME) $(BIN)/testreals
	-cmp testrealin.cbf testrealout.cbf
	$(TIME) $(BIN)/testflat
	-cmp testflatin.cbf testflatout.cbf
	$(TIME) $(BIN)/testflatpacked
	-cmp testflatpackedin.cbf testflatpackedout.cbf
ifneq ($(F90C),)
	echo testflatout.cbf | $(TIME) $(BIN)/test_xds_binary > test_xds_bin_testflatout.out
	-$(DIFF) test_xds_bin_testflatout.out test_xds_bin_testflatout_orig.out
	echo testflatpackedout.cbf | $(TIME) $(BIN)/test_xds_binary > test_xds_bin_testflatpackedout.out
	-$(DIFF) test_xds_bin_testflatpackedout.out test_xds_bin_testflatpackedout_orig.out
	echo testflatout.cbf | $(TIME) $(BIN)/test_fcb_read_image  > test_fcb_read_testflatout.out
	-$(DIFF) test_fcb_read_testflatout.out test_fcb_read_testflatout_orig.out
	echo testflatpackedout.cbf | $(TIME) $(BIN)/test_fcb_read_image > test_fcb_read_testflatpackedout.out
	-$(DIFF) test_fcb_read_testflatpackedout.out test_fcb_read_testflatpackedout_orig.out
endif
	$(TIME) $(BIN)/sauter_test
	$(TIME) $(BIN)/changtestcompression
	$(TIME) (export LD_LIBRARY_PATH=$(LIB);$(BIN)/tiff2cbf XRD1621.tif XRD1621.cbf)
	-$(DIFF) XRD1621.cbf XRD1621_orig.cbf
	$(TIME) $(BIN)/cif2cbf -I 4 -C 100. -L 0. -e n -c b -i XRD1621.cbf -o XRD1621_I4encbC100.cbf
	-$(DIFF) XRD1621_I4encbC100.cbf XRD1621_I4encbC100_orig.cbf

ifneq ($(F90C),)
extra_sigs_only:	$(BIN)/convert_image $(BIN)/convert_minicbf $(BIN)/cif2cbf $(BIN)/testcell \
	$(BIN)/testreals $(BIN)/testflat $(BIN)/testflatpacked \
	$(BIN)/test_xds_binary $(BIN)/test_fcb_read_image $(BIN)/convert_minicbf \
	$(BIN)/sauter_test $(BIN)/adscimg2cbf $(BIN)/cbf2adscimg $(BIN)/tiff2cbf \
	basic $(TESTINPUT_EXTRA) $(TESTOUTPUTSIGS)
else
extra_sigs_only:	$(BIN)/convert_image $(BIN)/convert_minicbf $(BIN)/cif2cbf $(BIN)/testcell \
	$(BIN)/testreals $(BIN)/testflat $(BIN)/testflatpacked \
	$(BIN)/convert_minicbf \
	$(BIN)/sauter_test $(BIN)/adscimg2cbf\
	basic $(TESTINPUT_EXTRA) $(TESTOUTPUTSIGS)
endif
	$(TIME) $(BIN)/cif2cbf -e hex -c none \
		makecbf.cbf cif2cbf_ehcn.cif
	$(TIME) $(BIN)/cif2cbf -e none -c packed \
		cif2cbf_ehcn.cif cif2cbf_encp.cbf; rm cif2cbf_ehcn.cif
	-cmp makecbf.cbf cif2cbf_encp.cbf
	$(TIME) $(BIN)/cif2cbf -i 9ins.cif -o 9ins.cbf
	-cmp 9ins.cif 9ins.cbf
	$(TIME) $(BIN)/convert_image -F example.mar2300 converted_flat.cbf
	-$(SIGNATURE) < converted_flat.cbf | $(DIFF) - converted_flat_orig.cbf$(SEXT); rm converted_flat.cbf
	$(TIME) $(BIN)/convert_image example.mar2300 converted.cbf
	-$(SIGNATURE) < converted.cbf | $(DIFF) - converted_orig.cbf$(SEXT); rm converted.cbf
	-$(TIME) $(BIN)/testcell < testcell.dat | \
	$(SIGNATURE) | $(DIFF) - testcell_orig.prt$(SEXT)
	$(TIME) $(BIN)/convert_image -F -d adscquantum315 mb_LP_1_001.img adscconverted_flat.cbf
	-$(SIGNATURE) < adscconverted_flat.cbf | $(DIFF) - adscconverted_flat_orig.cbf$(SEXT)
	$(TIME) $(BIN)/convert_image -d adscquantum315 mb_LP_1_001.img adscconverted.cbf
	-$(SIGNATURE) < adscconverted.cbf | $(DIFF) - adscconverted_orig.cbf$(SEXT); rm adscconverted.cbf
	$(TIME) $(BIN)/adscimg2cbf  --cbf_packed,flat mb_LP_1_001.img
	-$(SIGNATURE) < mb_LP_1_001.cbf | $(DIFF) - mb_LP_1_001_orig.cbf$(SEXT)
	mv mb_LP_1_001.cbf nmb_LP_1_001.cbf
	$(TIME) $(BIN)/cbf2adscimg nmb_LP_1_001.cbf
	-cmp nmb_LP_1_001.img mb_LP_1_001.img
	rm nmb_LP_1_001.cbf
	rm nmb_LP_1_001.img
	$(TIME) $(BIN)/convert_minicbf -d pilatus6m insulin_pilatus6m.cbf insulin_pilatus6mconverted.cbf
	-$(SIGNATURE) < insulin_pilatus6mconverted.cbf | $(DIFF) - insulin_pilatus6mconverted_orig.cbf$(SEXT); rm insulin_pilatus6mconverted.cbf
	$(TIME) $(BIN)/testreals
	-cmp testrealin.cbf testrealout.cbf
	$(TIME) $(BIN)/testflat
	-cmp testflatin.cbf testflatout.cbf
	$(TIME) $(BIN)/testflatpacked
	-cmp testflatpackedin.cbf testflatpackedout.cbf
ifneq ($(F90C),)
	echo testflatout.cbf | $(TIME) $(BIN)/test_xds_binary | \
	$(SIGNATURE) | $(DIFF) - test_xds_bin_testflatout_orig.out$(SEXT)
	echo testflatpackedout.cbf | $(TIME) $(BIN)/test_xds_binary | \
	$(SIGNATURE) | $(DIFF) - test_xds_bin_testflatpackedout_orig.out$(SEXT)
	echo testflatout.cbf | $(TIME) $(BIN)/test_fcb_read_image | \
	$(SIGNATURE) | $(DIFF) - test_fcb_read_testflatout_orig.out$(SEXT)
	echo testflatpackedout.cbf | $(TIME) $(BIN)/test_fcb_read_image | \
	$(SIGNATURE) | $(DIFF) - test_fcb_read_testflatpackedout_orig.out$(SEXT)
endif
	$(TIME) $(BIN)/sauter_test
	$(TIME) $(BIN)/tiff2cbf XRD1621.tif XRD1621.cbf
	$(TIME) $(BIN)/cif2cbf -I 4 -C 100. -L 0. -e n -c b -i XRD1621.cbf -o XRD1621_I4encbC100.cbf
	-$(SIGNATURE) < XRD1621.cbf | $(DIFF) - XRD1621_orig.cbf$(SEXT); rm XRD1621.cbf
	-$(SIGNATURE) < XRD1621_I4encbC100.cbf | $(DIFF) - XRD1621_I4encbC100_orig.cbf$(SEXT); rm XRD1621_I4encbC100.cbf
	@-rm -f adscconverted_flat.cbf
	@-rm -f $(TESTINPUT_BASIC) $(TESTINPUT_EXTRA) $(TESTOUTPUTSIGS)
	@-rm -f cif2cbf_packed.cbf    makecbf.cbf \
	cif2cbf_packed.cbf    img2cif_packed.cbf \
	cif2cbf_canonical.cbf img2cif_canonical.cbf
	@-rm -f testrealout.cbf testflatout.cbf testflatpackedout.cbf \
	cif2cbf_encp.cbf img2cif_canonical.cif img2cif_packed.cif 9ins.cbf


	
pycbftests:  $(PYCBF)/_pycbf.$(PYCBFEXT) 
	(cd $(PYCBF); python pycbf_test1.py)
	(cd $(PYCBF); python pycbf_test2.py)
	(cd $(PYCBF); python pycbf_test3.py)

javatests: $(BIN)/ctestcbf $(BIN)/testcbf.class $(SOLIB)/libcbf_wrap.so
	$(BIN)/ctestcbf > testcbfc.txt
	$(LDPREFIX) java -cp $(JCBF)/cbflib-$(VERSION).jar:$(BIN) testcbf > testcbfj.txt
	$(DIFF) testcbfc.txt testcbfj.txt

dectristests: $(BIN)/cbf_template_t $(DECTRIS_EXAMPLES)/cbf_test_orig.out
	(cd $(DECTRIS_EXAMPLES); ../../bin/cbf_template_t; diff -a -u cbf_test_orig.out cbf_template_t.out)

#
# Remove all non-source files
#
empty:
	@-rm -f  $(LIB)/*.o
	@-rm -f  $(LIB)/libcbf.a
	@-rm -f  $(LIB)/libfcb.a
	@-rm -f  $(LIB)/libimg.a
	@-rm -f  $(LIB)/_pycbf.$(PYCBFEXT)
	@-rm -f  $(PYCBF)/_pycbf.$(PYCBFEXT)
	@-rm -f  $(PYCBF)/build/*/_pycbf.$(PYCBFEXT)
	@-rm -f  $(PYCBF)/build/src/cbf_simple.o
	@-rm -f  $(PYCBF)/build/*/pycbf_wrap.o
	@-rm -rf  $(BIN)/adscimg2cbf*
	@-rm -rf  $(BIN)/cbf2adscimg*
	@-rm -rf  $(BIN)/makecbf*
	@-rm -rf  $(BIN)/img2cif*
	@-rm -rf  $(BIN)/cif2cbf*
	@-rm -rf  $(BIN)/convert_image*
	@-rm -rf  $(BIN)/convert_minicbf*
	@-rm -rf  $(BIN)/test_fcb_read_image*
	@-rm -rf  $(BIN)/test_xds_binary*
	@-rm -rf  $(BIN)/testcell*
	@-rm -rf  $(BIN)/cif2c*
	@-rm -rf  $(BIN)/testreals*
	@-rm -rf  $(BIN)/testflat*
	@-rm -rf  $(BIN)/testflatpacked*
	@-rm -rf  $(BIN)/cbf_template_t*
	@-rm -rf  $(BIN)/sauter_test*
	@-rm -rf  $(BIN)/arvai_test*
	@-rm -rf  $(BIN)/changtestcompression*
	@-rm -rf  $(BIN)/tiff2cbf*
	@-rm -f  makecbf.cbf
	@-rm -f  img2cif_packed.cif
	@-rm -f  img2cif_canonical.cif
	@-rm -f  img2cif_packed.cbf
	@-rm -f  img2cif_canonical.cbf
	@-rm -f  img2cif_raw.cbf
	@-rm -f  cif2cbf_packed.cbf
	@-rm -f  cif2cbf_canonical.cbf
	@-rm -f  converted.cbf
	@-rm -f  adscconverted.cbf
	@-rm -f  converted_flat.cbf
	@-rm -f  adscconverted_flat.cbf
	@-rm -f  adscconverted_flat_rev.cbf
	@-rm -f  mb_LP_1_001.cbf
	@-rm -f  cif2cbf_ehcn.cif
	@-rm -f  cif2cbf_encp.cbf
	@-rm -f  9ins.cbf
	@-rm -f  9ins.cif
	@-rm -f  testcell.prt
	@-rm -f  example.mar2300
	@-rm -f  converted_orig.cbf
	@-rm -f  adscconverted_orig.cbf
	@-rm -f  converted_flat_orig.cbf
	@-rm -f  adscconverted_flat_orig.cbf
	@-rm -f  adscconverted_flat_rev_orig.cbf
	@-rm -f  mb_LP_1_001_orig.cbf
	@-rm -f  insulin_pilatus6mconverted_orig.cbf
	@-rm -f  insulin_pilatus6mconverted.cbf
	@-rm -f  insulin_pilatus6m.cbf
	@-rm -f  testrealin.cbf
	@-rm -f  testrealout.cbf
	@-rm -f  testflatin.cbf
	@-rm -f  testflatout.cbf
	@-rm -f  testflatpackedin.cbf
	@-rm -f  testflatpackedout.cbf
	@-rm -f  CTC.cbf
	@-rm -f  test_fcb_read_testflatout.out
	@-rm -f  test_fcb_read_testflatpackedout.out
	@-rm -f  test_xds_bin_testflatpackedout.out
	@-rm -f  test_xds_bin_testflatout.out
	@-rm -f  test_fcb_read_testflatout_orig.out
	@-rm -f  test_fcb_read_testflatpackedout_orig.out
	@-rm -f  test_xds_bin_testflatpackedout_orig.out
	@-rm -f  test_xds_bin_testflatout_orig.out
	@-rm -f  mb_LP_1_001.img
	@-rm -f  9ins.cif
	@-rm -f  testcell_orig.prt
	@-rm -f  $(DECTRIS_EXAMPLES)/cbf_template_t.out
	@-rm -f  XRD1621.cbf
	@-rm -f  XRD1621_orig.cbf
	@-rm -f  XRD1621_orig.cbf
	@-rm -f  XRD1621_I4encbC100.cbf
	@-rm -f  $(SRC)/fcb_exit_binary.f90
	@-rm -f  $(SRC)/fcb_next_binary.f90
	@-rm -f  $(SRC)/fcb_open_cifin.f90
	@-rm -f  $(SRC)/fcb_packed.f90
	@-rm -f  $(SRC)/fcb_read_bits.f90
	@-rm -f  $(SRC)/fcb_read_image.f90
	@-rm -f  $(SRC)/fcb_read_xds_i2.f90
	@-rm -f  $(EXAMPLES)/test_fcb_read_image.f90
	@-rm -f  $(EXAMPLES)/test_xds_binary.f90
	@-rm -f  symlinksdone
	@-rm -f  $(TESTOUTPUT) *$(SEXT)
	@-rm -f  $(SOLIB)/*.o
	@-rm -f  $(SOLIB)/libcbf_wrap.so
	@-rm -f  $(SOLIB)/libjcbf.so
	@-rm -f  $(SOLIB)/libimg.so
	@-rm -f  $(SOLIB)/libfcb.so
	@-rm -rf $(JCBF)/org
	@-rm -f  $(JCBF)/*.java
	@-rm -f  $(JCBF)/jcbf_wrap.c
	@-rm -f  $(SRC)/cbf_wrap.c 
	@-rm -f  $(BIN)/ctestcbf $(BIN)/testcbf.class testcbfc.txt testcbfj.txt
	@-rm -rf  $(REGEX)
	@-rm -rf  $(TIFF)
	./.undosymlinks
	
#
# Remove temporary files
#
clean:	
	@-rm -f core 
	@-rm -f *.o
	@-rm -f *.u
#
# Restore to distribution state
#
distclean:	clean empty

#
# Create a Tape Archive for distribution
#

tar:   $(DOCUMENTS) $(SOURCE) $(SRC)/cbf.stx $(HEADERS) $(M4FILES)\
	 $(EXAMPLES) \
	 README.html README Makefile \
	 $(JPEGS)
	-/bin/rm -f CBFlib.tar*
	tar cvBf CBFlib.tar \
	 $(DOCUMENTS) $(SOURCE) $(SRC)/cbf.stx $(HEADERS) $(M4FILES)\
	 $(EXAMPLES) \
	 README.html README Makefile \
	 $(JPEGS)
	 gzip --best CBFlib.tar
'
