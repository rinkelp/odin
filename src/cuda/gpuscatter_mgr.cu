/*
This file implements a class that provides an interface for the GPU
scattering code (interface in gpuscatter.hh). It that takes data in on the 
cpu side, copies it to the gpu, and exposes functions that let you perform 
actions with the GPU.

This class will get translated into python via swig
*/


#include <stdio.h>
#include <assert.h>
#include <fstream>
#include <sstream>

#include <gpuscatter.cu>
#include <gpuscatter_mgr.hh>

using namespace std;


void deviceMalloc( void ** ptr, int bytes) {
    cudaError_t err = cudaMalloc(ptr, (size_t) bytes);
    assert(err == 0);
}


GPUScatter::GPUScatter (int device_id_,
                        int bpg_,      // <-- defines the number of rotations
            
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
    
    // check all input
    int z
    
    printf("qx:\n");
    for(z = 0; z < 5; z++) {
        printf("%f ", h_qx_[z]);
    }
    printf("rx:\n");
    for(z = 0; z < 5; z++) {
        printf("%f ", h_rx_[z]);
    }
    printf("rid:\n");
    for(z = 0; z < 5; z++) {
        printf("%f ", h_rid_[z]);
    }
    printf("cm:\n");
    for(z = 0; z < 5; z++) {
        printf("%f ", h_cm_[z]);
    }
    printf("rand1:\n");
    for(z = 0; z < 5; z++) {
        printf("%f ", h_rand1_[z]);
    }
    printf("outQ:\n");
    for(z = 0; z < 5; z++) {
        printf("%f ", h_outQ_[z]);
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
    
    // set the device
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        printf("Error setting device ID. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error synching device. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    // set some size parameters
    static const int tpb = 512;
    unsigned int nRotations = tpb*bpg;
    
    // compute the memory necessary to hold input/output
    const unsigned int nQ_size = nQ*sizeof(float);
    const unsigned int nAtoms_size = nAtoms*sizeof(float);
    const unsigned int nAtoms_idsize = nAtoms*sizeof(int);
    const unsigned int nRotations_size = nRotations*sizeof(float);
    const unsigned int cm_size = 9*numAtomTypes*sizeof(float);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error before device malloc. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // allocate memory on the board
    float *d_qx;        deviceMalloc( (void **) &d_qx, nQ_size);
    float *d_qy;        deviceMalloc( (void **) &d_qy, nQ_size);
    float *d_qz;        deviceMalloc( (void **) &d_qz, nQ_size);
    float *d_outQ;      deviceMalloc( (void **) &d_outQ, nQ_size);
    float *d_rx;        deviceMalloc( (void **) &d_rx, nAtoms_size);
    float *d_ry;        deviceMalloc( (void **) &d_ry, nAtoms_size);
    float *d_rz;        deviceMalloc( (void **) &d_rz, nAtoms_size);
    int   *d_id;        deviceMalloc( (void **) &d_id, nAtoms_idsize);
    float *d_cm;        deviceMalloc( (void **) &d_cm, cm_size);
    float *d_rand1;     deviceMalloc( (void **) &d_rand1, nRotations_size);
    float *d_rand2;     deviceMalloc( (void **) &d_rand2, nRotations_size);
    float *d_rand3;     deviceMalloc( (void **) &d_rand3, nRotations_size);
    
    // check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in device malloc. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // copy input/output arrays to board memory
    cudaMemcpy(d_qx, &h_qx[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qy, &h_qy[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qz, &h_qz[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outQ, &h_outQ[0], nQ_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rx, &h_rx[0], nAtoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ry, &h_ry[0], nAtoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rz, &h_rz[0], nAtoms_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_id, &h_id[0], nAtoms_idsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cm, &h_cm[0], cm_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand1, &h_rand1[0], nRotations_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand2, &h_rand2[0], nRotations_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rand3, &h_rand3[0], nRotations_size, cudaMemcpyHostToDevice);

    // check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in cuda memcpy. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // execute the kernel
    kernel<tpb> <<<bpg, tpb>>> (d_qx, d_qy, d_qz, d_outQ, nQ, d_rx, d_ry, d_rz, d_id, nAtoms, numAtomTypes, d_cm, d_rand1, d_rand2, d_rand3, nRotations);
    // kernel<tpb> <<<bpg, tpb>>> (d_qx, d_qy, d_qz, d_outQ, nQ, d_rx, d_ry, d_rz, d_id, nAtoms, numAtomTypes, d_rand1, d_rand2, d_rand3, nRotations);
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in kernel. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // retrieve the output off the board and back into CPU memory
    // copys the array to the output array passed as input
    cudaMemcpy(&h_outQ[0], d_outQ, nQ_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in memcpy from device --> host. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // free memory
    cudaFree(d_qx);
    cudaFree(d_qy);
    cudaFree(d_qz);
    cudaFree(d_rx);
    cudaFree(d_ry);
    cudaFree(d_rz);
    cudaFree(d_id);
    cudaFree(d_cm);
    cudaFree(d_rand1);
    cudaFree(d_rand2);
    cudaFree(d_rand3);
    cudaFree(d_outQ);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error freeing memory. CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    printf("output:\n");
    for(int x = 0; x < nQ; x++) {
        printf("%f ", h_outQ[x]);
    }

}

GPUScatter::~GPUScatter() {
    // destroy the class
}
