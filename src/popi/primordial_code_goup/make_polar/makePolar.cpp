#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <sstream>
#include "polarPixels.h"

using namespace std;

const float PI = 3.14159265;

void polarizationCorrection(vector<float>& polarPix,float Q,float Nphi,float pixelSize,float detDistance, float outOfPlane){
	int index(0);
	float dphi = 2*PI / Nphi;
	float theta = atan( (Q*pixelSize)/detDistance );
	float CosTheta = cos(theta);
	float SinTheta = sin(theta);
	float correction;
	float CosPhi;
	float SinPhi;
	for(float phi=0;phi < 2*PI; phi+=dphi){
		CosPhi = cos(phi);
		SinPhi = sin(phi);
		correction = outOfPlane*(1-SinTheta*SinTheta*CosPhi*CosPhi);
		correction += (1-outOfPlane)*(1-SinTheta*SinTheta*SinPhi*SinPhi);
		polarPix[ int(Q*Nphi) + index ] = polarPix[ int(Q*Nphi) + index ] / correction;
		index++;
		}
	}

void PolarPilatus::getRing(int Nphi_, float q_pix, vector<float>& IvsPhi )
{
  Nphi = Nphi_;

  cosPhi.clear();
  sinPhi.clear();
  for(int i=0; i < Nphi; i++)
  {
    float phi = float(i) * 2*PI/float(Nphi);
    cosPhi[i] = cos(phi);
    sinPhi[i] = sin(phi);
  }

  getPixelsAtQ( A, IvsPhi, 0, q_pix, x_center, y_center);
}


void PolarPilatus::nearest_multiple(int &n, int m)
{
  //adjusts the integer n so that it is envenly divisible by the integer m
  int remainder = n % m;
  if (remainder != 0)
	n += m - reminder;
}

vector<float> PolarPilatus::binPhi (int            numBins, int            samplesPerBin, int qpix, 
                      vector<float>& IvsPhi,  vector<float>& IvsPhi_binned)
// bins pixels azimuathally, forming rings of scattering. Accumulates these rings in IvsPhi_binned

{
  int i(0);
  while (i < numBins)
  {
    int j(0);
    float ave(0);
    while(j < samplesPerBin)
    {
      ave += IvsPhi[i*samplesPerBin + j];
      j   += 1;
    }
    ave = ave / float(samplesPerBin);
    IvsPhi_binned[numBins*qpix + i] = ave;
    i += 1;
  }
}

void PolarPilatus::InterpolateToPolar(float polarization, float qres, int Nphi_)
{

cout << "\n    Beginning full polar interpolation...\n";

// max Q in pixels units on the detector (with a 2 pixel cushion)
  float maxq_pix = floor( (float)Xdim/2)-2;
  if(Ydim < Xdim)
    maxq_pix = floor( (float)Ydim/2)-2;

// and how many bins this corresponds to in qres units
  int Nq(0);
  float maxq = sin( atan2( maxq_pix*pixSize, detDist ) / 2.)* 4. * M_PI /  wavelen;
  for(float q=0; q < maxq ; q += qres)
    Nq += 1;


// make a container for the pixels which are binned azimuthally...
// --> basically one ring per pixel unit radially. The radial binning will come next...
  vector<float> IvsPhi_binnedPhi (maxq_pix*Nphi_, 0);

  float q_pix(0);
  while(q_pix < maxq_pix)
  {
    int numPhiSamples = int(2*M_PI*q_pix); // Sample each ring at single-pixel resolution.
    if( numPhiSamples < Nphi_ )              // In case we are at very low q and numPhiSamples < Nphi,
  	numPhiSamples = Nphi_;              // then we force to sample Nphi times...
    nearest_multiple(numPhiSamples, Nphi_);  // Force numPhiSamples to be divisible by Nphi.
    vector<float> IvsPhi (numPhiSamples, 0);
    getRing( numPhiSamples, q_pix, IvsPhi);  // sample the ring at q_pix
    binPhi(Nphi_, numPhiSamples/Nphi_ , qpix, IvsPhi, IvsPhi_binnedPhi); // average the bins
    q_pix += 1;
  }

  Nphi = Nphi_; // safely reset the global variable Nphi; (getRing --> getPixelsAtQ mucks around with it a bit)

/*
  HERE WE BIN THE PIXELS RADIALLY ON THE DETECOTR, BUT IN RECIPROCAL SPACE UNITS.

  This is a bit tricky due to the non linearity in the q_pix <--> q_inv_ang relationship..
  --> The goal is to bin on an inv_angstrom scale. 
      The non-linearity means that a ring of width 0.02 inv_ang at low q contains fewer pixels radially
      than one at high q...

  q       is in inv_ang.
  q_stop  is in inv_ang.
  q_pix   is in pixel units.
  q_index is in qres units (e.g. 0.02 inverse angstroms).

  Open for suggestions on improving this section...
*/
  polar_pixels = new float[Nq*Nphi]; // this is the final polar interpolated image container
  for(int i=0;i < Nq*Nphi; i++)
    polar_pixels[i] = 0;
  
  q_pix        = 0;
  float q      = sin( atan2( q_pix * pixsize, detdist) / 2.)* 4. * M_PI/wavelen;
  float q_stop = dq_iA; // we will stop averaging once we reach q_stop, then we will do q_stop += qres
  int q_index  = 0;


  while (q_index < Nq )
  {
  float counts(0);
  while (q < q_stop)
  {
    for(int i=0;i < Nphi; i++)
      polar_pixels[q_index*Nphi + i] += IvsPhi_binnedPhi[q_pix*numPhiBins + i];
    q_pix+=1;
    q = sin( atan2( q_pix * pixSize, detDist ) / 2. ) * 4. * M_PI / wavelen;
    counts += 1;
    if (q==maxq_pix) // nasty, but the last ring is usually not important (furthest out on detector)
      break;
  }
  q_stop += qres;
  for(int i=0;i < Nphi;i++)
    polar_pixels[q_index*NPhi + i] = polar_pixels[q_index*Nphi + i] / counts;
  q_index += 1;
  }

}
