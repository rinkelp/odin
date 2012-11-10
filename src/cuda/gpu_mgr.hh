
/* Header file for gpu_mgr.cu, GPUScatter class */

class GPUScatter {
    
    // declare variables
    int bpg_;      // <-- defines the number of rotations

    int nQ_;
    int* h_qx_;    // size: nQ
    int* h_qy_;    // size: nQ
    int* h_qz_;    // size: nQ

    int nAtoms_;
    int* h_rx_;    // size: nAtoms
    int* h_ry_;    // size: nAtoms
    int* h_rz_;    // size: nAtoms
    int* h_id_;    // size: nAtoms

    int* h_rand1_; // size: nRotations
    int* h_rand2_; // size: nRotations
    int* h_rand3_; // size: nRotations

    int* h_outQ_;  // size: nQ (OUTPUT)

public:
  /* By using the swig default names INPLACE_ARRAY1, DIM1 in the header
     file (these aren't the names in the implementation file), we're giving
     swig the info it needs to cast to and from numpy arrays.
     
     If instead the constructor line said
       GPUAdder(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

  GPUAdder( int bpg_,      // <-- defines the number of rotations
            
            int nQ_,
            int* h_qx_,    // size: nQ
            int* h_qy_,    // size: nQ
            int* h_qz_,    // size: nQ

            int nAtoms_,
            int* h_rx_,    // size: nAtoms
            int* h_ry_,    // size: nAtoms
            int* h_rz_,    // size: nAtoms
            int* h_id_,    // size: nAtoms

            int* h_rand1_, // size: nRotations
            int* h_rand2_, // size: nRotations
            int* h_rand3_, // size: nRotations

            int* h_outQ_,  // size: nQ (OUTPUT)
           );
  void run();                              // does operation inplace on the GPU
  void retreive();                         // gets results back from GPU
  ~GPUAdder();                             // destructor
};
