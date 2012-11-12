/* -*- C -*-  (not really, but good for syntax highlighting) */

%module gpuscatter
/* This is the swig "interface file" which gives instructions to swig
   on how to turn the class declared in manager.hh into the python module
   "gpuadder"

   The key thing that we want it to accomplish is automatic conversion of
   arrays from numpy (python) into CUDA/C++ (simply pointers and lengths).
   Provided that we give swig proper instructions, either by special naming
   of the variables in the header file (manager.hh) or by a instruction in this
   file (line 30), swig can do the numpy<->c++ conversion seamlessly.
*/

%{
    #define SWIG_FILE_WITH_INIT
    #include "gpuscatter_mgr.hh"
%}

// swig doesn't know about numpy by default, so we need to give it an extra numpy interface
// file that I downloaded from http://docs.scipy.org/doc/numpy/reference/swig.interface-file.html

%include "numpy.i"
%include <typemaps.i>

%init %{
    import_array();
%}

// %apply int* INPUT {int* bpg_, int* nQ_, int* nAtoms_, int* numAtomTypes_};

%apply (int DIM1, float* IN_ARRAY1) {(int nQ0, float* h_qx_), 
                                     (int nQ1, float* h_qy_),
                                     (int nQ2, float* h_qz_),
                                     (int nAtoms0, float* h_rx_),
                                     (int nAtoms1, float* h_ry_),
                                     (int nAtoms2, float* h_rz_),
                                     (int ncm, float* h_cm_),
                                     (int nRot0, float* h_rand1_),
                                     (int nRot1, float* h_rand2_),
                                     (int nRot2, float* h_rand3_)};
                                     
%apply (int DIM1, float* ARGOUT_ARRAY1) {(int nQ3, float* h_outQ_)};

/* if instead the names of the pointers were not the standard ones, this
   type of translation would be necessary.
   http://www.scipy.org/Cookbook/SWIG_NumPy_examples */

%include "gpuscatter_mgr.hh"
