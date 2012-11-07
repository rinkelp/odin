

// Functions for simulating the scattered intensity from an emsemble of randomly
// oriented molecules.
// 
// Functionality is included for calcuating the scattering profile from an 
// ensemble of different configurations -- such as a Boltzmann ensemble 
// described by molecular simulation.
// 
// Assumes that file io will be performed by python, this function wraps into
// python.
// 
// 
// TJL 11/5/12


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// pseudo-random number generator TJL NEED TO UPDATE
float rfloat() {
      float r = (float)rand()/(float)RAND_MAX;
      return r;
}

// random quaternion generator
void rquaternion( double *q ) {
    
    float s, sig1, sig2, theta1, theta2, w, x, y, z;
    
    s = rfloat();
    sig1 = sqrt(s);
    sig2 = sqrt(1.0 - s);
    
    theta1 = 2.0 * M_PI * rfloat();
    theta2 = 2.0 * M_PI * rfloat();
    
    w = cos(theta2) * sig2;
    x = sin(theta1) * sig1;
    y = cos(theta1) * sig1;
    z = sin(theta2) * sig2;
    
    q[0] = w;
    q[1] = x;
    q[2] = y;
    q[3] = z;
}



float form_factor(float q, int atomz) {
    
    float fi;
    float qo = (q*q) / (4 * 4 * M_PI * M_PI);
    
    if ( atomz == 1 ) {
        fi = 0.493002*exp(-10.5109*qo);
        fi+= 0.322912*exp(-26.1257*qo);
        fi+= 0.140191*exp(-3.14236*qo);
        fi+= 0.040810*exp(-57.7997*qo);
        fi+= 0.003038;
    }
    else if ( atomz == 8) {
        fi = 3.04850*exp(-13.2771*qo);
        fi+= 2.28680*exp(-5.70110*qo);
        fi+= 1.54630*exp(-0.323900*qo);
        fi+= 0.867000*exp(-32.9089*qo);
        fi+= 0.2508;
    }
    else if ( atomz == 26) {
        fi = 11.7695*exp(-4.7611*qo);
        fi+= 7.35730*exp(-0.307200*qo);
        fi+= 3.52220*exp(-15.3535*qo);
        fi+= 2.30450*exp(-76.8805*qo);
        fi+= 1.03690;
    }
    else if ( atomz == 79) {
        fi = 16.8819*exp(-0.4611*qo);
        fi+= 18.5913*exp(-8.6216*qo);
        fi+= 25.5582*exp(-1.4826*qo);
        fi+= 5.86*exp(-36.3956*qo);
        fi+= 12.0658;
    }
    // else approximate with Nitrogen
    else
        fi = 12.2126*exp(-0.005700*qo);
        fi+= 3.13220*exp(-9.89330*qo);
        fi+= 2.01250*exp(-28.9975*qo);
        fi+= 1.16630*exp(-0.582600*qo);
        fi+= -11.529;
    return fi;
}


// void image(const float *xyzlist, int traj_length, int num_atoms, 
//             int molecules_per_shot, int num_q, const double *qvec, 
//             double* results) {
// 
//     /* Calculates the scattering intensity for an ensemble of molecules for each
//      * value of q requested.
//      *
//      * Pulls uniformly from xyzlist to obtain samples of the xyz coordinates.
//      */
//     
//     int i, j, k;
//     double qx, qy, qz, rx, ry, rz;
//     double *results_ptr;
//     const float *frame; /* TJL find out what this does! */
//     
//     // loop over all molecules
//     #pragma omp parallel for default(none) shared(results, xyzlist, traj_length, num_atoms, molecules_per_shot, num_q, q_mag, q_ang) private(i, j)
//     for (i = 0; i < molecules_per_shot; i++) {
//         
//         frame = xyzlist + num_atoms * 3 * i; /* which frame - should be random */
//         
//         // perform a random rotation
//         
//         
//         
//         // loop over all q-vectors
//         for (j = 0; j < num_q; j++) {
//             
//         results_ptr = results + i;
//         
//         qx = *(qvec + j * 3);
//         qy = *(qvec + j * 3 + 1);
//         qz = *(qvec + j * 3 + 2);
//         
//             // loop over all atoms
//             for (k = 0; k < num_atoms; k++) {
//                 
//                 rx = *(frame + k * 3);
//                 ry = *(frame + k * 3 + 1);
//                 rz = *(frame + k * 3 + 2);
//                 
//                 dot = rx*qx + ry*qz + rz*qz;
//                 
//                 double fi = 1.0 /* TJL change later */
//                 double F = cabs( fi * cexp( I * dot ) )
//                 
//                 *results_ptr += F*F;
//             }
//         }
//     }
// }
          
      
main() {
    
    printf("test...\n");
    
    int d;
    double q[4];
    
    // seed random number generator
    srand( time(NULL) );
    
    rquaternion(q);
    
    for( d = 0; d < 4; d++) {
        printf("%f\n", q[d]);
    }
    
}
               
               
               



