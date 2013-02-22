#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include "bicub.h"

using namespace std;

const float PI = 3.14159265;

vector<float> PolarPilatus::getRing(int Nphi_, float q_pix )
{
// detector params

// experimental params
//float polarization = 0.99; // fraction in out-of-plane with synchrtron

  //int Nq =  int(maxQ/qRes); // num q bins (rounds down)

// make a polar grid
  Nphi = Nphi_;

  cosPhi.clear();
  sinPhi.clear();
  for(int i=0; i < Nphi; i++)
  {
    float phi = float(i) * 2*PI/float(Nphi);
    cosPhi[i] = cos(phi);
    sinPhi[i] = sin(phi);
  }

// array for storing final output
  vector<float> polarPix (Nphi,0);
  getPixelsAtQ( A, polarPix, 0, q_pix, x_center, y_center);
  return polarPix;
}
