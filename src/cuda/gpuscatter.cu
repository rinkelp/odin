/*! YTZ 20121106 */
#include <stdio.h>
#include <stdint.h>

// ============================================================================
// IF YOU ARE HERE BECAUSE YOUR num_atom_types WAS TOO BIG...
// then increment the number below according to your needs. 

#define MAX_NUM_TYPES 10

// ============================================================================


// warning: this code is not safe due to reduction if total # of threads != multiple of
// blockSize ... too lazy to add in ifs for now 
// todo: add in ifs and while loops for > 67million
void __device__ generate_random_quaternion(float r1, float r2, float r3,
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

__device__ double atomicAdd(double* address, double val) {
    double old = *address, assumed;
    do{
        assumed = old;
        old =__longlong_as_double(atomicCAS((unsigned long long int*)address,
            __double_as_longlong(assumed),
            __double_as_longlong(val + assumed)));
    }
    while(assumed != old);
    return old;
}


void __device__ rotate(float x, float y, float z,
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

  //float cc0 = c0*bb0 - c1*bb1 - c2*bb2 - c3*bb3;
    float cc1 = c0*bb1 + c1*bb0 + c2*bb3 - c3*bb2;
    float cc2 = c0*bb2 - c1*bb3 + c2*bb0 + c3*bb1;
    float cc3 = c0*bb3 + c1*bb2 - c2*bb1 + c3*bb0;   

    ox = cc1;
    oy = cc2;
    oz = cc3;

}

// blockSize = tpb, templated in case we need to use a faster reduction
// method later. 
template<unsigned int blockSize>
void __global__ kernel(float const * const __restrict__ q_x, 
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
                       unsigned int numRotations) {

    // shared array for block-wise reduction
    __shared__ float sdata[blockSize];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    // blank-out reduction buffer. 
    sdata[tid] = 0;
    __syncthreads();
    
    while(gid < numRotations) {
       
        // determine the rotated locations
        float rand1 = randN1[gid]; 
        float rand2 = randN2[gid]; 
        float rand3 = randN3[gid]; 

        // rotation quaternions
        float q0, q1, q2, q3;
        generate_random_quaternion(rand1, rand2, rand3, q0, q1, q2, q3);

        // for each q vector
        for(int iq = 0; iq < nQ; iq++) {
            float qx = q_x[iq];
            float qy = q_y[iq];
            float qz = q_z[iq];

            // workspace for cm calcs -- static size, but hopefully big enough
            float formfactors[MAX_NUM_TYPES];

            // accumulant
            float2 Qsum;
            Qsum.x = 0;
            Qsum.y = 0;
     
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
            // bottle-necked by this currently. 
            for(int a = 0; a < numAtoms; a++) {

                // get the current positions
                float rx = r_x[a];
                float ry = r_y[a];
                float rz = r_z[a];
                int   id = r_id[a];
                float ax, ay, az;

                rotate(rx, ry, rz, q0, q1, q2, q3, ax, ay, az);
                float qr = ax*qx + ay*qy + az*qz;

                fi = formfactors[id];
                
                // tjl for debug
                Qsum.x += __sinf(qr);
                Qsum.y += __cosf(qr);
                
                // Qsum.x += fi*__sinf(qr);
                // Qsum.y += fi*__cosf(qr);
            } // finished one molecule.
            
            float fQ = (Qsum.x*Qsum.x + Qsum.y*Qsum.y) / numRotations;  
            sdata[tid] = fQ;
            __syncthreads();

            // Todo: quite slow reduction but correct, speed up reduction later if becomes bottleneck!
            for(unsigned int s=1; s < blockDim.x; s *= 2) {
                if(tid % (2*s) == 0) {
                    sdata[tid] += sdata[tid+s];
                }
                __syncthreads();
            }
            if(tid == 0) {
                atomicAdd(outQ+iq, sdata[0]); 
            } 
        }

        // blank out reduction buffer.
        sdata[tid] = 0;

        // syncthreads are important here!
        __syncthreads();

        // offset by total working threads across all blocks. 
        gid += gridDim.x * blockDim.x;
    }

}
