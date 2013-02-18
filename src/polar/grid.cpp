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

using namespace std;

class PolarScatter_SM
{

  void generate_random_quaternion(
                 float r1, float  r2, float r3,
                float &q1, float &q2, float &q3, float &q4);

  void rotate(float x, float y, float z,
              float b0, float b1, float b2, float b3,
              float &ox, float &oy, float &oz);

  void sleep(int mseconds);

public:
  PolarScatter_SM (string in_file,int Nphi,int n_rotations,float qres, float wavelen, string qVal);
 ~PolarScatter_SM ();

};



PolarScatter_SM::PolarScatter_SM 
    (string in_file,int Nphi,int n_rotations,
     float qres, float wavelen, string qVal) {

hid_t   in_file_id = H5Fopen(in_file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
hid_t   h5_file_id = H5Fcreate( (char*)( in_file +".ring").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

if(in_file_id < 0)
  cout <<"ERROR: cannot locate file " << in_file << endl;

hid_t    data_id_xyza = H5Dopen(in_file_id,"xyza/",H5P_DEFAULT);
hsize_t  size         = H5Dget_storage_size( data_id_xyza ); // size in bytes of molecule info
int      numAtoms     = (int)size/32; // 4 floats per atom
float   *xyza         = new float[numAtoms*4]; // x,y,z,id for each atom
// read the coordinate info into the array xyza
herr_t   read_xyza    = H5Dread(data_id_xyza, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, xyza);

//hid_t    space_xyza  = H5Dget_space (data_id_xyza);

//int   Nphi        = atoi(argv[2]); // number of bins around diffraction ring
float deltaPhi    = 2*M_PI / Nphi;   // spacing between scattering points around the ring
//int   n_rotations = atoi(argv[3]); // number of random rotations
//float qres        = atof(argv[4]); // units of q in inverse angstroms
//float wavelen     = atof(argv[5]); // wavelength of experiment in angstroms

// close coordinate file
H5Fclose(in_file_id);

// define an array data spaces
hsize_t  rank1[1],rank2[2]; // one array element per dimension of data space
rank1[0] = 1; // single float data space
rank2[0] = n_rotations; // first dimension of 2D data space
rank2[1] = Nphi; // second dimension of 2D data space
hid_t  space_single = H5Screate_simple(1,rank1,NULL); // single float space
hid_t  space_rings  = H5Screate_simple(2,rank2,NULL); // 2D data space (num_rotations * Nphi)

/*
for each q value we will iterate over random orientations, each
time computing the azimuthal scattering signal at that q.
We will then save this information in a 2D array (n_rotaions*Nphi) and
then save that to hdf5 
*/

// save the qres
hid_t   data_q_res = H5Dcreate1(h5_file_id, "/qres", H5T_NATIVE_FLOAT, space_single , H5P_DEFAULT);
herr_t  err_q_res  = H5Dwrite(data_q_res, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &qres);
H5Dclose(data_q_res); // close dataset


sleep(500); // delay to avoid rotation repeats
srand (time(NULL)); // intitalize random seed
float * quats = new float[4*n_rotations];
vector<string> shot_names;
for (int im=0; im < n_rotations; im ++)
{
  float rand1   = (float)rand()/RAND_MAX;
  float rand2   = (float)rand()/RAND_MAX;
  float rand3   = (float)rand()/RAND_MAX;
  
  generate_random_quaternion(
    rand1,       rand2,         rand3, 
    quats[4*im], quats[4*im+1], quats[4*im+2], quats[4*im+3]);
}



cout << "\n    Starting simulation.\n    Estimated time to complete: " 
     << 153./1000. * float(1) *  (float)n_rotations / 3600.  << " hours...\n\n";
         
vector<float> THETA,Q; vector<string> QSTRING;
for(int i=0;i<1;i++)
{
  float q = atof(qVal.c_str() )*qres;
  Q.push_back( q );
  float sin_theta = wavelen*q/4/M_PI;
  THETA.push_back( asin(sin_theta) );
  QSTRING.push_back( qVal );
}

float ax,ay,az; // atomic positions (used later)

for(int i=0;i < Q.size();i++)
{
  float theta = THETA[i]; // scattering angle divided by 2
  float cosTheta = cos(theta);
  float q = Q[i]; if(q==0)q+=0.001; //magnitude of q vector
    
  float qz = q*sin(theta); // z-component of q vector
  string qName = QSTRING[i]; // name of scattering output file
  

  cout << "    ---> Computing the scattering ring at q = "<< q << " Ang^-1 ...\n";
  
  hid_t  rings_data_id = H5Dcreate1(h5_file_id,("/ring_"+qName).c_str() , H5T_NATIVE_FLOAT, space_rings , H5P_DEFAULT);
  
  float * rings = new float[n_rotations*Nphi];
  for (int im=0; im < n_rotations; im ++)
  {
    float phi(0);
    int phiIndex(0);
    while (phi < 2*M_PI)
    {
      float qx = q*cosTheta*cos(phi);
      float qy = q*cosTheta*sin(phi);
      float QsumR(0),QsumI(0); // sum1 = real part, sum2 = im part
      for(int a = 0; a < numAtoms; a+=4)
      {
        // we are rotating more times than necessary, but this is better for mem management. 
        rotate( xyza[a],     xyza[a+1],     xyza[a+2],
                quats[4*im], quats[4*im+1], quats[4*im+2], quats[4*im+3],
                ax,          ay,            az );
	float phase = ax*qx + ay*qy + az*qz;
	//float phase = xyza[a]*qx + xyza[a+1]*qy + xyza[a+2]*qz;
	QsumR       +=  xyza[a+3]*cos(phase); // add in crommer man functionality
	QsumI       +=  xyza[a+3]*sin(phase); // --> something like CM( xyza[a+3] )
      }
      rings[im*Nphi + phiIndex] = QsumR*QsumR + QsumI*QsumI;
      phi += deltaPhi;
      phiIndex += 1;
    }
  }
//save the rings to hdf
  herr_t  write_rings = H5Dwrite(rings_data_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rings);
  H5Dclose(rings_data_id);
}

// close some ish
H5Sclose(space_single);
H5Sclose(space_rings);
H5Fclose(h5_file_id);

}


void PolarScatter_SM::generate_random_quaternion
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


void PolarScatter_SM::rotate
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

void PolarScatter_SM::sleep(int mseconds) { usleep(mseconds * 1000); }

PolarScatter_SM::~PolarScatter_SM(){}

int main(){

PolarScatter_SM pssm ("gold_11k.hdf",360,10,0.02,0.7293,"133");

return 0;}
