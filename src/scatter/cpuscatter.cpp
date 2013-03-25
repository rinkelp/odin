
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <iostream>


#ifdef NO_OMP
   #define omp_get_thread_num() 0
#else
   #include <omp.h>
#endif

#include "cpuscatter.hh"

using namespace std;

/*! TJL 2012 */

#define MAX_NUM_TYPES 10

void generate_random_quaternion(float r1, float r2, float r3,
                float &q1, float &q2, float &q3, float &q4) {
    
    float s, sig1, sig2, theta1, theta2, w, x, y, z;
    
    s = r1;
    sig1 = sqrt(s);
    sig2 = sqrt(1.0 - s);
    
    theta1 = 2.0 * M_PI * r2;
    theta2 = 2.0 * M_PI * r3;
    
    w = cos(theta2) * sig2;
    x = sin(theta1) * sig1;
    y = cos(theta1) * sig1;
    z = sin(theta2) * sig2;
    
    q1 = w;
    q2 = x;
    q3 = y;
    q4 = z;
}


void rotate(float x, float y, float z,
            float b0, float b1, float b2, float b3,
            float &ox, float &oy, float &oz) {

    // x,y,z      -- float vector
    // b          -- quaternion for rotation
    // ox, oy, oz -- rotated float vector
    
    float a0 = 0;
    float a1 = x;
    float a2 = y;
    float a3 = z;

    float c0 = b0*a0 - b1*a1 - b2*a2 - b3*a3;
    float c1 = b0*a1 + b1*a0 + b2*a3 - b3*a2;
    float c2 = b0*a2 - b1*a3 + b2*a0 + b3*a1;
    float c3 = b0*a3 + b1*a2 - b2*a1 + b3*a0;   

    float bb0 = b0;
    float bb1 = -b1;
    float bb2 = -b2;
    float bb3 = -b3;

    float cc1 = c0*bb1 + c1*bb0 + c2*bb3 - c3*bb2;
    float cc2 = c0*bb2 - c1*bb3 + c2*bb0 + c3*bb1;
    float cc3 = c0*bb3 + c1*bb2 - c2*bb1 + c3*bb0;   

    ox = cc1;
    oy = cc2;
    oz = cc3;

}

// "kernel" is the function that computes the scattering intensities
void kernel( float const * const __restrict__ q_x, 
             float const * const __restrict__ q_y, 
             float const * const __restrict__ q_z, 
             float *outQ, // <-- not const 
             int   const nQ,
             float const * const __restrict__ r_x, 
             float const * const __restrict__ r_y, 
             float const * const __restrict__ r_z,
             int   const * const __restrict__ r_id, 
             int   const numAtoms, 
             int   const numAtomTypes,
             float const * const __restrict__ cromermann,
             float const * const __restrict__ randN1, 
             float const * const __restrict__ randN2, 
             float const * const __restrict__ randN3,
             int   const n_rotations,
             int   const finite_photons,
             int   const * const __restrict__ n_photons ) {

    // if doing finite photons, we need some memory allocated to store results
    int ndQ;
    if ( finite_photons == 1 ) {
        ndQ = nQ;
    } else {
        ndQ = 1; // don't allocate a lot of mem we wont use
    }
    srand(randN1[0]);                   // init rand seed, using the first rand
                                        // number makes the code deterministic
                                        // for the purposes of testing
    int * discrete_outQ = new int[ndQ]; // new array for finite photon output
    for( int iq = 0; iq < ndQ; iq++ ) {
        discrete_outQ[iq] = 0;
    }

    // main loop -- over molecules
    for( int im = 0; im < n_rotations; im++ ) {
        
        // keep track of the total scattered intensity
        float outQ_sum = 0.0;
       
        // determine the rotated locations
        float rand1 = randN1[im]; 
        float rand2 = randN2[im]; 
        float rand3 = randN3[im]; 

        // rotation quaternions
        float q0, q1, q2, q3;
        generate_random_quaternion(rand1, rand2, rand3, q0, q1, q2, q3);

        // for each q vector
        #pragma omp parallel for shared(outQ, q0, q1, q2, q3)
        for( int iq = 0; iq < nQ; iq++ ) {
            float qx = q_x[iq];
            float qy = q_y[iq];
            float qz = q_z[iq];

            // workspace for cm calcs -- static size, but hopefully big enough
            float formfactors[MAX_NUM_TYPES];

            // accumulant
            float Qsumx; Qsumx = 0;
            float Qsumy; Qsumy = 0;

            // Cromer-Mann computation, precompute for this value of q
            float mq = qx*qx + qy*qy + qz*qz;
            float qo = mq / (16*M_PI*M_PI); // qo is (sin(theta)/lambda)^2
            float fi;
            
            // for each atom type, compute the atomic form factor f_i(q)
            for (int type = 0; type < numAtomTypes; type++) {
            
                // scan through cromermann in blocks of 9 parameters
                int tind = type * 9;
                fi =  cromermann[tind]   * exp(-cromermann[tind+4]*qo);
                fi += cromermann[tind+1] * exp(-cromermann[tind+5]*qo);
                fi += cromermann[tind+2] * exp(-cromermann[tind+6]*qo);
                fi += cromermann[tind+3] * exp(-cromermann[tind+7]*qo);
                fi += cromermann[tind+8];
                
                formfactors[type] = fi; // store for use in a second
            }

            // for each atom in molecule
            for( int a = 0; a < numAtoms; a++ ) {

                // get the current positions
                float rx = r_x[a];
                float ry = r_y[a];
                float rz = r_z[a];
                int   id = r_id[a];
                float ax, ay, az;

                rotate(rx, ry, rz, q0, q1, q2, q3, ax, ay, az);
                float qr = ax*qx + ay*qy + az*qz;

                fi = formfactors[id];
                Qsumx += fi*sinf(qr);
                Qsumy += fi*cosf(qr);
            } // end loop over atoms
                        
            // add the output to the total intensity array
            #pragma omp critical
            outQ[iq] += (Qsumx*Qsumx + Qsumy*Qsumy);
            outQ_sum += outQ[iq];
            
        } // end loop over q
        
        // discrete photon statistics, if requested
        if ( finite_photons == 1 ) {
                
            float rp;
            float cum_q_sum;
        
            // for each photon, draw a rand to put it in a pixel
            for ( int p = 0; p < n_photons[im]; p++ ) {
            
                rp = (float)rand()/(float)RAND_MAX * outQ_sum;
                
                cum_q_sum = 0.0;
                int iq = 0;
            
                while ( cum_q_sum < rp ) {
                    cum_q_sum += outQ[iq];
                    iq++;
                }
                discrete_outQ[iq] += 1;
            }
        } // end finite photons
        
    } // end loop over rotations
    
    // now cp the results in discrete_outQ to outQ and free memory
    // a bit silly, but is only one loop :/
    if ( finite_photons == 1 ) {
        for( int iq = 0; iq < nQ; iq++ ) {
            outQ[iq] = discrete_outQ[iq];
        }
    }
    delete [] discrete_outQ;
    
} // end kernel fxn


CPUScatter::CPUScatter( int    nQ_, // scattering vectors
                        float* h_qx_,
                        float* h_qy_,
                        float* h_qz_,
                
                        // atomic positions, ids
                        int    nAtoms_,
                        float* h_rx_,
                        float* h_ry_,
                        float* h_rz_,
                        int*   h_id_,

                        // cromer-mann parameters
                        int    nCM_,
                        float* h_cm_,

                        // random numbers for rotations
                        int    nRot_,
                        float* h_rand1_,
                        float* h_rand2_,
                        float* h_rand3_,
                        
                        // finite photons
                        int    finite_photons_,
                        int*   n_photons_,

                        // output
                        float* h_outQ_ ) {
                                
    // unpack arguments
    
    n_rotations = nRot_;
    nQ = nQ_;
    h_qx = h_qx_;
    h_qy = h_qy_;
    h_qz = h_qz_;

    nAtoms = nAtoms_;
    int numAtomTypes = nCM_ / 9;
    h_rx = h_rx_;
    h_ry = h_ry_;
    h_rz = h_rz_;
    h_id = h_id_;

    h_cm = h_cm_;

    h_rand1 = h_rand1_;
    h_rand2 = h_rand2_;
    h_rand3 = h_rand3_;
    
    finite_photons = finite_photons_;
    n_photons = n_photons_;

    h_outQ = h_outQ_;

    // execute the kernel
    kernel(h_qx, h_qy, h_qz, h_outQ, nQ, h_rx, h_ry, h_rz, h_id, nAtoms, 
           numAtomTypes, h_cm, h_rand1, h_rand2, h_rand3, n_rotations, 
           finite_photons, n_photons);
}

CPUScatter::~CPUScatter() {
    // destroy the class
}
