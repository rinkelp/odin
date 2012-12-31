/* -*- C -*-  (not really, but good for syntax highlighting) */

%module cpuscatter

%{
    #define SWIG_FILE_WITH_INIT
    #include "cpuscatter.h"
%}


%include "numpy.i" // from http://docs.scipy.org/doc/numpy/reference/swig.interface-file.html

%init %{
    import_array();
%}


%apply (int DIM1, float* IN_ARRAY1) {(int nQx_, float* h_qx_), 
                                     (int nQy_, float* h_qy_),
                                     (int nQz_, float* h_qz_),
                                     (int nAtomsx_, float* h_rx_),
                                     (int nAtomsy_, float* h_ry_),
                                     (int nAtomsz_, float* h_rz_),
                                     (int nCM_, float* h_cm_),
                                     (int nRot1_, float* h_rand1_),
                                     (int nRot2_, float* h_rand2_),
                                     (int nRot3_, float* h_rand3_)}
%apply (int DIM1, int* IN_ARRAY1)  {(int nAtomsid_, int* h_id_)}
                                   
%apply (int DIM1, float* ARGOUT_ARRAY1) {(int nQout_, float* h_outQ_)};

%include "cpuscatter.h"
