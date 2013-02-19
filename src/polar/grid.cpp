#include <cstdlib>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#include <sstream>
#include <math.h>
#include <sstream>
#include <unistd.h>
#include <time.h>

// external libs
#include <hdf5.h>
#include <grid.h>

using namespace std;

RingScatter::RingScatter (
     string in_file, int   Nphi_,    int    n_rotations_,
     float  qres_,   float wavelen_, string qstring_) 
{

  Nphi        = Nphi_;        // num azimuthal points around each ring on detector
  qres        = qres_;        // units in inverse angstroms (each qval is an integer in these units)
  wavelen     = wavelen_;     // wavelen of beam in angstroms
  qstring        = qstring_;        // string of integers (space delimited) telling RingScatter::Scatter() where to compute the scattering
                              // --> each integer is in units of qres
  n_rotations = n_rotations_; // number of random orientations to sample
  
  rings       = new float[n_rotations*Nphi];


/*
  OPEN THE INPUT FILE CONTAING THE COORDINATE INFO.
  THIS FILE WILL BE OVERWRITTEN SO THAT THE NEW INFORMATION CAN BE SAVED.
*/

  h5_file_id = H5Fopen(in_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if(h5_file_id < 0)
    cout <<"ERROR: cannot locate file " << in_file << endl;
  
/*
  READ THE ATOMIC COORDINATE INFO
*/
  hid_t    data_id_xyza  = H5Dopen(h5_file_id,"/xyza",H5P_DEFAULT); // open coordinate dataset
  hid_t    space_xyza    = H5Dget_space (data_id_xyza);             // data space of coordinate dataset
  hsize_t  size          = H5Dget_storage_size( data_id_xyza );     // size in bytes (BYTES) of coordinate info
  numAtoms               = (int)size/16;                            // 4 floats per atom
  float   *xyza          = new float[numAtoms*4];                   // x,y,z,id for each atom
  herr_t   read_xyza     = H5Dread(data_id_xyza, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, xyza);
  for(int i=0;i < numAtoms*4;i++)
    XYZA.push_back(xyza[i]);     // initialize the coordinate info
  H5Dclose(data_id_xyza);        //close the dataset to keep things clean
  H5Fclose(h5_file_id);          // close the file so we can re-open it

  cout << "\n    Found coordinate information for " << numAtoms << " atoms ...\n";

/*
  OVERWRITE EXISTING COORDINATE FILE AND SAVE THE COORDINATE INFO
*/

  h5_file_id           = H5Fcreate( in_file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);       // open/overwrite file
  data_id_xyza         = H5Dcreate1(h5_file_id,"/xyza",H5T_NATIVE_FLOAT,space_xyza,H5P_DEFAULT);     // create dataset
  herr_t write_xyza    = H5Dwrite(data_id_xyza ,H5T_NATIVE_FLOAT,H5S_ALL,H5S_ALL,H5P_DEFAULT, xyza); // write coordinate info
  H5Sclose(space_xyza);   // close dataspace to keep things clean
  H5Dclose(data_id_xyza); // close dataset to keep things clean
  delete [] xyza;         // free the heap
  
/*
  DEFINE DATA SPACES FOR THE HDF5 DATASETS
  THESE ARE USED LATER TO SAVE INFO TO HDF5 FORMAT
*/
  hsize_t        rank1[1],rank2[2];               // one array element per dimension of data space
  rank1[0]     = 1;                               // single float data space
  rank2[0]     = n_rotations;                     // first dimension of 2D data space
  rank2[1]     = Nphi;                            // second dimension of 2D data space
  space_single = H5Screate_simple(1,rank1,NULL);  // single float space
  space_rings  = H5Screate_simple(2,rank2,NULL);  // 2D data space (num_rotations * Nphi)

/*
  SAVE THE PARAMETER qres TO HDF5
*/
  h5_ring_group_id   = H5Gcreate(h5_file_id, "/rings",H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
  hid_t   data_q_res = H5Dcreate1(h5_ring_group_id, "qres", H5T_NATIVE_FLOAT, space_single , H5P_DEFAULT);
  herr_t  err_q_res  = H5Dwrite(data_q_res, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &qres);
  H5Dclose(data_q_res); // close dataset


/*
  INITIALIZE RANDOM QUATERNIONS FOR ROTATIONS
*/

  sleep(500); // delay to avoid rotation repeats
  srand (time(NULL)); // intitalize random seed
  for(int i=0;i < 4*n_rotations;i++)
    quats.push_back(0);
  for (int im=0; im < n_rotations; im ++)
  {
    float rand1   = (float)rand()/RAND_MAX;
    float rand2   = (float)rand()/RAND_MAX;
    float rand3   = (float)rand()/RAND_MAX;
  
    generate_random_quaternion(
      rand1,       rand2,         rand3, 
      quats[4*im], quats[4*im+1], quats[4*im+2], quats[4*im+3]);
  }


/*
  INITIALIZE SCATTERING DETECTOR COORDINATES (RINGS IN POLAR COORDINATES)
*/
  split_string(qstring,' ',QVALS);
  for(int i=0;i<QVALS.size();i++)
  {
    float q         = atof(QVALS[i].c_str() )*qres;
    float sin_theta = wavelen*q/4/M_PI;
    Q.push_back( q );
    THETA.push_back( asin(sin_theta) );
  }


}

void RingScatter::Scatter()
{
/*
  for each q value we will iterate over random orientations, each
  time computing the azimuthal scattering signal at that q.
  We will then save this information in a 2D array (n_rotaions*Nphi) and
  then save that to hdf5 
*/

  cout << "\n    Starting simulation.\n    (Estimated time to complete: " 
       << 69./1000. * float(1) *  (float)n_rotations / 3600.  << " hours)\n\n";



  float ax,ay,az; // atomic positions (used later)

  for(int i=0;i < Q.size();i++)
  {
    float theta    = THETA[i];               // scattering angle divided by 2
    float cosTheta = cos(theta);
    float q        = Q[i]; if(q==0)q+=0.001; //magnitude of q vector
    
    float qz       = q*sin(theta);           // z-component of q vector

    cout << "    ---> Computing the scattering ring at q = "<< q << " Ang^-1 ...\n";
  
    hid_t  rings_data_id = H5Dcreate1(h5_ring_group_id,("ring_"+QVALS[i]).c_str() , H5T_NATIVE_FLOAT, space_rings , H5P_DEFAULT);
  

    for (int im=0; im < n_rotations; im ++)
    {
      float  phi(0);
      int    phiIndex(0);
      while (phiIndex < Nphi)
      {
        float qx = q*cosTheta*cos(phi);
        float qy = q*cosTheta*sin(phi);
        float QsumR(0),QsumI(0); // sum1 = real part, sum2 = im part
        for(int a = 0; a < numAtoms; a+=4)
        {
          // we are rotating more times than necessary, but this is better for mem management. 
          rotate( XYZA[a],     XYZA[a+1],     XYZA[a+2],
                  quats[4*im], quats[4*im+1], quats[4*im+2], quats[4*im+3],
                  ax,          ay,            az );
	  float phase = ax*qx + ay*qy + az*qz;
	  QsumR      +=  XYZA[a+3]*cos(phase); // add in crommer man functionality
	  QsumI      +=  XYZA[a+3]*sin(phase); // --> something like CM( xyza[a+3] )
        }
        rings[int( im*Nphi + phiIndex )] = QsumR*QsumR + QsumI*QsumI;
        phi += 2*M_PI/float(Nphi);
        phiIndex += 1;
      }
    }
//save the rings to hdf
    herr_t  write_rings = H5Dwrite(rings_data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rings);
    H5Dclose(rings_data_id);
  }

}



void RingScatter::generate_random_quaternion
	       (float  r1, float  r2, float  r3,
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


void RingScatter::rotate
	   (float x,   float   y, float   z,
            float b0,  float  b1, float  b2, float b3,
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

void RingScatter::split_string(const string &s, char delim, vector<string> &elems)
{
  istringstream ss(s);
  string item;
  while( getline(ss,item,delim) )
    elems.push_back(item) ;
}

void RingScatter::sleep(int mseconds) { usleep(mseconds * 1000); }


RingScatter::~RingScatter()
{
  delete [] rings;
  H5Sclose(space_single);
  H5Sclose(space_rings);
  H5Fclose(h5_file_id);

}




int main(){

RingScatter pssm ("gold_11k.hdf5",360,1,0.02,0.7293,"133 154");
pssm.Scatter();


return 0;}
