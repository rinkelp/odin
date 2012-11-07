/*! YTZ 20121106 */
#include <stdio.h>
#include <stdint.h>

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

// figure out mirrors...
/*
float __device__ fastsin(float x) {

    // move x to between -pi/2 and pi/2


    float a; // between 0 and pi/2

    // slower than store multiple copies of a*a, a*a*a, etc. but saves registers
    float r = a - (a*a*a)/(3*2) + (a*a*a*a*a)/(5*4*3*2) - (a*a*a*a*a*a*a)/(7*6*5*4*3*2);

    float f;
    
    f = (a < 0) ? a : -a;

    return f;
}
*/

void __device__ rotate(float x, float y, float z,
                       float b0, float b1, float b2, float b3,
                       float &ox, float &oy, float &oz) {

    float a0 = 0;
    float a1 = x;
    float a2 = y;
    float a3 = z;

    float c0 = a0*b0 - a1*b1 - a2*b2 - a3*b3;
    float c1 = a0*b1 + a1*b0 + a2*b3 - a3*b2;
    float c2 = a0*b2 - a1*b3 + a2*b0 + a3*b1;
    float c3 = a0*b3 + a1*b2 - a2*b1 + a3*b0;   

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


template<unsigned int blockSize>
void __global__ kernel(float const * const __restrict__ q_x, 
                       float const * const __restrict__ q_y, 
                       float const * const __restrict__ q_z, 
                       float *outQ, // <-- not const 
                       int   const nQ,
		               float const * const __restrict__ r_x, 
                       float const * const __restrict__ r_y, 
                       float const * const __restrict__ r_z,
		               int   const * const __restrict__ atomicIdentities, 
                       int   const numAtoms, 
                       float const * const __restrict__ randN1, 
                       float const * const __restrict__ randN2, 
                       float const * const __restrict__ randN3) {
    // shared array for block-wise reduction
    __shared__ float sdata[blockSize];
    
    int tid = threadIdx.x;
	int gid = blockIdx.x*blockDim.x + threadIdx.x;

    // determine the rotated locations
    float rand1 = randN1[gid]; 
    float rand2 = randN2[gid]; 
    float rand3 = randN3[gid]; 

    // rotation quaternions
    float q0, q1, q2, q3;
    generate_random_quaternion(rand1, rand2, rand3, q0, q1, q2,q3);

    // for each q vector
    for(int iq = 0; iq < nQ; iq++) {
        float qx = q_x[iq];
        float qy = q_y[iq];
        float qz = q_z[iq];
        float mq = qx*qx+qy*qy+qz*qz;
        float qo = mq / (4*4*M_PI*M_PI);
        //accumulant
        float2 Qsum;
        Qsum.x = 0;
        Qsum.y = 0;
        // for each atom in molecule
        for(int a = 0; a < numAtoms; a++) {
            // calculate fi
            float fi = 0;
            int atomicNumber = atomicIdentities[a];
            // if H
            if(atomicNumber == 1) {
                fi = 0.493002*exp(-10.5109*qo);
                fi+= 0.322912*exp(-26.1257*qo);
                fi+= 0.140191*exp(-3.14236*qo);
                fi+= 0.040810*exp(-57.7997*qo);
                fi+= 0.003038;
            // if C
            } /*else if(atomicNumber == 6) {
                // to do.
            // if O
            } else if(atomicNumber == 8) {
                fi = 3.04850*exp(-13.2771*qo);
                fi+= 2.28680*exp(-5.70110*qo);
                fi+= 1.54630*exp(-0.323900*qo);
                fi+= 0.867000*exp(-32.9089*qo);
                fi+= 0.2508;
            // if N 
            } else if(atomicNumber == 7) {
                fi = 12.2126*exp(-0.005700*qo);
                fi+= 3.13220*exp(-9.89330*qo);
                fi+= 2.01250*exp(-28.9975*qo);
                fi+= 1.16630*exp(-0.582600*qo);
                fi+= -11.529;
            // if Fe
            } else if(atomicNumber == 26) {
                fi = 11.7695*exp(-4.7611*qo);
                fi+= 7.35730*exp(-0.307200*qo);
                fi+= 3.52220*exp(-15.3535*qo);
                fi+= 2.30450*exp(-76.8805*qo);
                fi+= 1.03690;
            // if Au
            } */ else if(atomicNumber == 79) {
                fi = 16.8819*exp(-0.4611*qo);
                fi+= 18.5913*exp(-8.6216*qo);
                fi+= 25.5582*exp(-1.4826*qo);
                fi+= 5.86*exp(-36.3956*qo);
                fi+= 12.0658; 
            // else default to N
            }  /* else {
                fi = 12.2126*exp(-0.005700*qo);
                fi+= 3.13220*exp(-9.89330*qo);
                fi+= 2.01250*exp(-28.9975*qo);
                fi+= 1.16630*exp(-0.582600*qo);
                fi+= -11.529;
            } */

            // get the current positions
            float rx = r_x[a];
            float ry = r_y[a];
            float rz = r_z[a];
            float ax, ay, az;

            rotate(rx, ry, rz, q0, q1, q2, q3, ax, ay, az);
            
            float qr = ax*qx + ay*qy + az*qz;

            Qsum.x += fi*__sinf(qr);
            Qsum.y += fi*__cosf(qr);
            
        } // finished one molecule.

        float fQ = Qsum.x*Qsum.x + Qsum.y*Qsum.y;  
        //printf("tid: %d, fQ: %f\n", tid, fQ);

        sdata[tid] = fQ;
        __syncthreads();

        // Todo: quite slow - speed up reduction later!
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
}

__global__ void randTest(float *a) {
    int gid = blockIdx.x*blockDim.x + threadIdx.x;

    int tt = __cosf(gid);
    int yy = __sinf(gid);

    a[gid] = tt;
    a[gid/2] = yy;
}
