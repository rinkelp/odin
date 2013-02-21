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

int nearest_multiple(int &n, int m)
{//adjusts the integer n so that it is envenly divisible by the integer m
  int remainder = n % m;
  if (lastBin != 0)
	n += m - reminder;
}

void PolarPilatus::InterpolateToPolar(float polarization, float qres, int Nphi_)
{

  Nphi = Nphi_;

// max Q in pixels units on the detector (with a 2 pixel cushion)
  float maxq_pix = floor( (float)Xdim/2)-2;
  if(Ydim < Xdim)
    maxq_pix = floor( (float)Ydim/2)-2;

// and how many bins does this correspond to in qres units
  int Nq(0);
  float maxq = sin( atan2( maxq_pix*pixSize, detDist ) / 2.)* 4. * M_PI /  wavelen;
  for(float q=0; q < maxq ; q += qres)
    Nq += 1;

  vector<float> polarPix (polN,0);

  int numPhiSamples(X/2.);
  vector<float> IvsPhi_binnedPhi;
  float Q(0);
  while(Q < maxQ)
  {
    float circum(2*PI*Q);
    if (circum > numPhiSamples)
  	numPhiSamples = circum;
    getPixelsAtQ_bicubic_Bin(A,IvsPhi_binnedPhi,Q,numPhiSamples,a,b,numPhiBins);
    Q += 1;
  }

cout << "here" << endl;
//	bin Q
Q = 0;
Q_iA = sin( atan2(Q*pixSize,detDist) / 2)*(4*PI/wavelen);
float Q_iAStop = dQ_iA;
int indQ(0);
while (indQ < numQBins ){
	vector<float> qBin (numPhiBins,0);
	float counts(0);
	while (Q_iA < Q_iAStop){
		for(int i=0;i < numPhiBins;i++)
			qBin[i] += IvsPhi_binnedPhi[int(Q)*numPhiBins + i];
		Q+=1;
		Q_iA = sin( atan2(Q*pixSize,detDist) / 2)*(4*PI/wavelen);
		counts += 1;
		if (Q==maxQ)
			break;
		}
	Q_iAStop += dQ_iA;
	for(int i=0;i < numPhiBins;i++){
		qBin[i] = qBin[i] / counts;
		polarPix[indQ*numPhiBins + i] = qBin[i];
		}
	indQ += 1;
	}
cout << indQ << endl;

cout << "here" << endl;
float * newData = new float[polN];
for(int i=0;i < polN;i++){
	newData[i] = polarPix[i];
	}
string outFileName = string(argv[n]).substr( 0, string(argv[n]).length()-4 )+"-pol.bin";
cout << outFileName << endl;
FILE * outFile = fopen(outFileName.c_str(),"w");

fwrite(newHeader,1,1024,outFile);
fwrite(newData,4,polN,outFile);

delete [] newData;

fclose(outFile);

n++;

return 0;}
