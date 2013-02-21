#include <vector>

using namespace std;

class PolarPilatus
{
  int Xdim,Ydim;
  int Nphi;
  float pixsize,detdist,wavelen;

  vector<float> cosPhi,sinPhi,A;

  float F( float ar[],int i, int j );
  float EvaluateIntensity(float coef[], float x , float y);
  void bicubicCoefficients(float * I);
  float aveNo0(vector<float>& ar);
  void getPixelsAtQ(vector<float>& IvsPhi, int q_index, float q, float a, float b);

public:
  PolarPilatus(int Xdim_,      int Ydim_,      float * binData, 
               float detdist_, float pixsize_, float wavelen_);
  void Center(float qMin, float qMax, float center_res, int Nphi_, float size);
 ~PolarPilatus();
};
