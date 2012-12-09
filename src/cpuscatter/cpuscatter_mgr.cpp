#include <stdio.h>
#include <assert.h>

#include <cpuscatter.cu>
#include <cpuscatter_mgr.hh>

CPUScatter::CPUScatter( int n_rotations_,
            
                        // scattering q-vectors
                        int    nQx_,
                        float* h_qx_,
                        int    nQy_,
                        float* h_qy_,
                        int    nQz_,
                        float* h_qz_,
                
                        // atomic positions, ids
                        int    nAtomsx_,
                        float* h_rx_,
                        int    nAtomsy_,
                        float* h_ry_,
                        int    nAtomsz_,
                        float* h_rz_,
                        int    nAtomsid_,
                        int*   h_id_,

                        // cromer-mann parameters
                        int    nCM_,
                        float* h_cm_,

                        // random numbers for rotations
                        int    nRot1_,
                        float* h_rand1_,
                        int    nRot2_,
                        float* h_rand2_,
                        int    nRot3_,
                        float* h_rand3_,

                        // output
                        int    nQout_,
                        float* h_outQ_
                        ) {
    
    // sometimes SWIG fucks up and retains non-zero values in the output
    // array, so we have to initialize it
    int z;
    //printf("\noutQ:\n");
    for(z = 0; z < nQout_; z++) {
        //printf("%f ", h_outQ_[z]);
        h_outQ_[z] = 0.0;
        //printf("%f \n", h_outQ_[z]);
    }

    /* All arguments consist of 
     *   (1) a float pointer to the beginning of the array to be passed
     *   (2) ints representing the size of each array
     */

    // many of the arrays above are 1D arrays that should be the same len
    // due to the SWIG wrapping, however, we had to pass each individually
    // so now check that they are, in fact, the correct dimension
    assert( nQx_ == nQy_ );
    assert( nQx_ == nQz_ );
    assert( nQx_ == nQout_ );
    
    assert( nAtomsx_ == nAtomsy_ );
    assert( nAtomsx_ == nAtomsz_ );
    assert( nAtomsx_ == nAtomsid_ );
    
    assert( nRot1_ == nRot2_ );
    assert( nRot1_ == nRot3_ );    
    assert( bpg_ * 512 == nRot1_ );
    
    // unpack arguments
    device_id = device_id_;
    bpg = bpg_;

    nQ = nQx_;
    h_qx = h_qx_;
    h_qy = h_qy_;
    h_qz = h_qz_;

    nAtoms = nAtomsx_;
    numAtomTypes = nCM_ / 9;
    h_rx = h_rx_;
    h_ry = h_ry_;
    h_rz = h_rz_;
    h_id = h_id_;

    h_cm = h_cm_;

    h_rand1 = h_rand1_;
    h_rand2 = h_rand2_;
    h_rand3 = h_rand3_;

    h_outQ = h_outQ_;
        
    // compute the memory necessary to hold input/output
    const unsigned int nQ_size = nQ*sizeof(float);
    const unsigned int nAtoms_size = nAtoms*sizeof(float);
    const unsigned int nAtoms_idsize = nAtoms*sizeof(int);
    const unsigned int nRotations_size = nRotations*sizeof(float);
    const unsigned int cm_size = 9*numAtomTypes*sizeof(float);

    // allocate memory on the board
    // float *d_qx;        deviceMalloc( (void **) &d_qx, nQ_size);
    // float *d_qy;        deviceMalloc( (void **) &d_qy, nQ_size);
    // float *d_qz;        deviceMalloc( (void **) &d_qz, nQ_size);
    // float *d_outQ;      deviceMalloc( (void **) &d_outQ, nQ_size);
    // float *d_rx;        deviceMalloc( (void **) &d_rx, nAtoms_size);
    // float *d_ry;        deviceMalloc( (void **) &d_ry, nAtoms_size);
    // float *d_rz;        deviceMalloc( (void **) &d_rz, nAtoms_size);
    // int   *d_id;        deviceMalloc( (void **) &d_id, nAtoms_idsize);
    // float *d_cm;        deviceMalloc( (void **) &d_cm, cm_size);
    // float *d_rand1;     deviceMalloc( (void **) &d_rand1, nRotations_size);
    // float *d_rand2;     deviceMalloc( (void **) &d_rand2, nRotations_size);
    // float *d_rand3;     deviceMalloc( (void **) &d_rand3, nRotations_size);

    // execute the kernel
    kernel(h_qx, h_qy, h_qz, h_outQ, nQ, h_rx, h_ry, h_rz, h_id, nAtoms, numAtomTypes, h_cm, h_rand1, h_rand2, h_rand3, nRotations);

    // free memory
    // cudaFree(d_qx);
    // cudaFree(d_qy);
    // cudaFree(d_qz);
    // cudaFree(d_rx);
    // cudaFree(d_ry);
    // cudaFree(d_rz);
    // cudaFree(d_id);
    // cudaFree(d_cm);
    // cudaFree(d_rand1);
    // cudaFree(d_rand2);
    // cudaFree(d_rand3);
    // cudaFree(d_outQ);

CPUScatter::~CPUScatter() {
    // destroy the class
}