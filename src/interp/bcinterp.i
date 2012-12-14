/* -*- C -*-  (not really, but good for syntax highlighting) */

%module bcinterp

%{
    #define SWIG_FILE_WITH_INIT
    #include "bcinterp.hh"
%}


%include "numpy.i" // from http://docs.scipy.org/doc/numpy/reference/swig.interface-file.html

%init %{
    import_array();
%}


%apply (int DIM1, double* IN_ARRAY1) {(int Nvals, double* vals),
                                     (int dim_xa, double* xa),
                                     (int dim_ya, double* ya)};
                                   
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int dim_za, double* za)};

%include "bcinterp.hh"
