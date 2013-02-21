#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <math.h>

//external libs
#include <popi.h>


using namespace std;


PolarPilatus::PolarPilatus(int   Xdim_,     int Ydim_,       float * binData,
                           float detdist_, float pixsize_, float wavelen_)
{
  Xdim    = Xdim_;    // pixel units
  Ydim    = Ydim_;    // pixel units
  detdist = detdist_; // meters
  pixsize = pixsize_; // meters
  wavelen = wavelen_; // angstroms

  bicubicCoefficients(binData);


  for(int i=0; i < Nphi; i++)
  {
    float phi = float(i) * 2*M_PI/float(Nphi);
    cosPhi.push_back (cos(phi) );
    sinPhi.push_back( sin(phi) );
  }

}


float PolarPilatus::F(float ar[],int i,int j){
	float val = ar[j*Xdim+i];
	return val;
	}

float PolarPilatus::EvaluateIntensity(float coef[],float x,float y){
                float a00=coef[0];
                float a10=coef[1];
                float a20=coef[2];
                float a30=coef[3];
                float a01=coef[4];
                float a11=coef[5];
                float a21=coef[6];
                float a31=coef[7];
                float a02=coef[8];
                float a12=coef[9];
                float a22=coef[10];
                float a32=coef[11];
                float a03=coef[12];
                float a13=coef[13];
                float a23=coef[14];
                float a33=coef[15];
		float interpI = a00 + a10*x + a20*x*x + a30*x*x*x;
                interpI+= a01*y + a11*x*y + a21*x*x*y + a31*x*x*x*y;
                interpI+= a02*y*y + a12*x*y*y + a22*x*x*y*y + a32*x*x*x*y*y;
                interpI+= a03*y*y*y + a13*x*y*y*y + a23*x*x*y*y*y + a33*x*x*x*y*y*y;
                /*
                interpI = 0;
                aij=0
                j = 0
                while j <=3:
                        i = 0
                        while i <=3:
                                interpI += coef[aij]*pow(x,i)*pow(y,j)
                                aij += 1
                                i+=1
                        j+=1
                */
                return interpI;
		}

void  PolarPilatus::bicubicCoefficients( float *I ){
	const int N(Xdim*Ydim);

	float * dIdx = new float[N];
	float * dIdy = new float[N];
	float * dIdxdy = new float[N];
	float dx,dy,dxdy;

	int y(1);
	int x;
	while( y < Ydim-1){
		x=1;
		while (x<Xdim-1){
			dx = ( F(I,x+1,y) - F(I,x-1,y) ) / 2.0;
			dy = ( F(I,x,y+1) - F(I,x,y-1) ) / 2.0;
			dxdy = ( F(I,x+1,y+1) - F(I,x+1,y-1) - F(I,x-1,y+1) + F(I,x-1,y-1) ) / 4.0;
			dIdx[y*Xdim + x] = dx;
			dIdy[y*Xdim + x] = dy;
			dIdxdy[y*Xdim + x] = dxdy;
			/*
			if (F(I,x,y) != 0 && F(I,x+1,y) == 0){
				dx = F(I,x,y);
				dxdy = ( F(I,x,y+1) - F(I,x,y-1) - F(I,x-1,y+1) + F(I,x-1,y-1) ) / 4.0;
				}
			else if (F(I,x,y) == 0 && F(I,x+1,y) != 0){
				dx = F(I,x+1,y);
				dxdy = ( F(I,x+1,y+1) - F(I,x+1,y-1) - F(I,x-1,y+1) + F(I,x-1,y-1) ) / 4.0;
				}
			else if (F(I,x,y) != 0 && F(I,x,y+1) == 0){
				dy = F(I,x,y);
				dxdy = ( F(I,x+1,y) - F(I,x+1,y-1) - F(I,x-1,y) + F(I,x-1,y-1) ) / 4.0;
				}
			else if (F(I,x,y) == 0 && F(I,x,y+1) != 0){
				dy = F(I,x,y+1);
				dxdy = ( F(I,x+1,y+1) - F(I,x+1,y-1) - F(I,x-1,y+1) + F(I,x-1,y-1) ) / 4.0;
				}
			*/
			x += 1;
			}
		y += 1;
		}

	// top row (no corners)
	x = 1;
	while (x < Xdim-1){
		dx= dIdx[Xdim + x];
		dy= dIdy[Xdim + x];
		dxdy= dIdxdy[Xdim + x];
		dIdx[x] = dx;
		dIdy[x] = dy;
		dIdxdy[x] = dxdy;
		x += 1;
		}
	// do the bottom row (no corners)
	x = 1;
	while(x < Xdim-1){
		dx= dIdx[Xdim*(Ydim-2) + x];
		dy= dIdy[Xdim*(Ydim-2) + x];
		dxdy= dIdxdy[Xdim*(Ydim-2) + x];
		dIdx[Xdim*(Ydim-1) + x] = dx;
		dIdy[Xdim*(Ydim-1) + x] = dy;
		dIdxdy[Xdim*(Ydim-1) + x] = dxdy;
		x += 1;
		}

	// do the left side (no corners)
	y = 1;
	while(y < Ydim-1){
		dx= dIdx[Xdim*y + 1];
		dy= dIdy[Xdim*y + 1];
		dxdy= dIdxdy[Xdim*y + 1];
		dIdx[Xdim*y] = dx;
		dIdy[Xdim*y] = dy;
		dIdxdy[Xdim*y] = dxdy;
		y += 1;
		}
	// do the right side (no corners)
	y = 1;
	while(y < Ydim-1){
		dx= dIdx[Xdim*y + Xdim-2];
		dy= dIdy[Xdim*y + Xdim-2];
		dxdy= dIdxdy[Xdim*y + Xdim-2];
		dIdx[Xdim*y + Xdim-1] = dx;
		dIdy[Xdim*y + Xdim-1] = dy;
		dIdxdy[Xdim*y + Xdim-1] = dxdy;
		y += 1;
		}

	//do the top,left
	dx= dIdx[1];
	dy= dIdy[Xdim];
	dxdy = dIdxdy[Xdim+1];
	dIdx[0] = dx;
	dIdy[0] = dy;
	dIdxdy[0] = dxdy;

	//do the top,right
	dx = dIdx[Xdim-2];
	dy = dIdy[2*Xdim-1];
	dxdy = dIdxdy[2*Xdim-2];
	dIdx[Xdim-1] = dx;
	dIdy[Xdim-1] = dy;
	dIdxdy[Xdim-1] = dxdy;

	//do the bottom,left
	dx = dIdx[Xdim*(Ydim-1)+1];
	dy = dIdy[Xdim*(Ydim-2)];
	dxdy = dIdxdy[Xdim*(Ydim-2)+1];
	dIdx[Xdim*(Ydim-1)] = dx;
	dIdy[Xdim*(Ydim-1)] = dy;
	dIdxdy[Xdim*(Ydim-1)] = dxdy;

	//do the bottom,right
	dx = dIdx[Xdim*Ydim-2];
	dy = dIdy[Xdim*(Ydim-1)-1];
	dxdy = dIdxdy[Xdim*(Ydim-1)-2];
	dIdx[Xdim*Ydim-1] = dx;
	dIdy[Xdim*Ydim-1] = dy;
	dIdxdy[Xdim*Ydim-1] = dxdy;
/*
	// correct for gaps
	y = 1;
	while (y < Ydim-1){
		x = 1;
		while(x < Xdim-1){
			if (I[y*Xdim + x] == 0 && I[y*Xdim + x - 1] != 0){
				dIdx[y*Xdim + x] = dIdx[y*Xdim + x -1];
				dIdxdy[y*Xdim + x] = dIdxdy[(y+1)*Xdim + x -1];
				}
			else if (I[y*Xdim + x] == 0 && I[(y-1)*Xdim + x] != 0){
				dIdy[y*Xdim + x] = dIdy[(y-1)*Xdim + x];
				dIdxdy[y*Xdim + x] = dIdxdy[(y-1)*Xdim + x - 1];
				}
			else if (I[y*Xdim + x] == 0 && I[y*Xdim + x + 1] != 0){
				dIdx[y*Xdim + x] = dIdx[y*Xdim + x +1];
				dIdxdy[y*Xdim + x] = dIdxdy[(y+1)*Xdim + x +1];
				}
			else if (I[y*Xdim + x] == 0 && I[(y+1)*Xdim + x] != 0){
				dIdy[y*Xdim + x] = dIdy[(y+1)*Xdim + x];
				dIdxdy[y*Xdim + x] = dIdxdy[(y+1)*Xdim + x + 1];
				}
			x += 1;
			}
		y += 1;
		}
*/


	cout << "\n    --> Calculating interpolation coefficients ...\n";

	y=0;
	while(y < Ydim-1){
		x=0;
		while(x<Xdim-1){
			float a00 = F(I,x,y);
			float a10 = F(dIdx,x,y);
			float a20 = -3*F(I,x,y) + 3*F(I,x+1,y) - 2*F(dIdx,x,y) - F(dIdx,x+1,y);
			float a30 = 2*F(I,x,y) - 2*F(I,x+1,y) + F(dIdx,x,y) + F(dIdx,x+1,y);

			float a01 = F(dIdy,x,y);
			float a11 = F(dIdxdy,x,y);
			float a21 = -3*F(dIdy,x,y) + 3*F(dIdy,x+1,y) - 2*F(dIdxdy,x,y) - F(dIdxdy,x+1,y);
			float a31 = 2*F(dIdy,x,y) - 2*F(dIdy,x+1,y) + F(dIdxdy,x,y) + F(dIdxdy,x+1,y);

			float a02 = -3*F(I,x,y) + 3*F(I,x,y+1) - 2*F(dIdy,x,y) - F(dIdy,x,y+1);
			float a12 = -3*F(dIdx,x,y) + 3*F(dIdx,x,y+1) - 2*F(dIdxdy,x,y) - F(dIdxdy,x,y+1);
			float a22 = 9*F(I,x,y) - 9*F(I,x+1,y) - 9*F(I,x,y+1) + 9*F(I,x+1,y+1);
			a22+= 6*F(dIdx,x,y) + 3*F(dIdx,x+1,y) - 6*F(dIdx,x,y+1) - 3*F(dIdx,x+1,y+1);
			a22+= 6*F(dIdy,x,y) - 6*F(dIdy,x+1,y) + 3*F(dIdy,x,y+1) - 3*F(dIdy,x+1,y+1);
			a22+= 4*F(dIdxdy,x,y) + 2*F(dIdxdy,x+1,y) + 2*F(dIdxdy,x,y+1) + F(dIdxdy,x+1,y+1);
			float a32 = -6*F(I,x,y) + 6*F(I,x+1,y) + 6*F(I,x,y+1) - 6*F(I,x+1,y+1);
			a32+= -3*F(dIdx,x,y) - 3*F(dIdx,x+1,y) + 3*F(dIdx,x,y+1)+ 3*F(dIdx,x+1,y+1);
			a32+= -4*F(dIdy,x,y) + 4*F(dIdy,x+1,y) - 2*F(dIdy,x,y+1) + 2*F(dIdy,x+1,y+1);
			a32+= -2*F(dIdxdy,x,y) - 2*F(dIdxdy,x+1,y) -F(dIdxdy,x,y+1) -F(dIdxdy,x+1,y+1);

			float a03 = 2*F(I,x,y) - 2*F(I,x,y+1) + F(dIdy,x,y) + F(dIdy,x,y+1);
			float a13 = 2*F(dIdx,x,y) - 2*F(dIdx,x,y+1) + F(dIdxdy,x,y) + F(dIdxdy,x,y+1);
			float a23 = -6*F(I,x,y) + 6*F(I,x+1,y) + 6*F(I,x,y+1) - 6*F(I,x+1,y+1);
			a23+= -4*F(dIdx,x,y) -2*F(dIdx,x+1,y) + 4*F(dIdx,x,y+1) + 2*F(dIdx,x+1,y+1);
			a23+= -3*F(dIdy,x,y) + 3*F(dIdy,x+1,y) - 3*F(dIdy,x,y+1) + 3*F(dIdy,x+1,y+1);
			a23+= -2*F(dIdxdy,x,y) - F(dIdxdy,x+1,y) - 2*F(dIdxdy,x,y+1) - F(dIdxdy,x+1,y+1);
			float a33 = 4*F(I,x,y) - 4*F(I,x+1,y) - 4*F(I,x,y+1) + 4*F(I,x+1,y+1);
			a33+= 2*F(dIdx,x,y) + 2*F(dIdx,x+1,y) - 2*F(dIdx,x,y+1) - 2*F(dIdx,x+1,y+1);
			a33+= 2*F(dIdy,x,y) - 2*F(dIdy,x+1,y) + 2*F(dIdy,x,y+1) - 2*F(dIdy,x+1,y+1);
			a33+= F(dIdxdy,x,y) + F(dIdxdy,x+1,y) + F(dIdxdy,x,y+1) + F(dIdxdy,x+1,y+1);
			
			A.push_back(a00);
			A.push_back(a10);
			A.push_back(a20);
			A.push_back(a30);
			A.push_back(a01);
			A.push_back(a11);
			A.push_back(a21);
			A.push_back(a31);
			A.push_back(a02);
			A.push_back(a12);
			A.push_back(a22);
			A.push_back(a32);
			A.push_back(a03);
			A.push_back(a13);
			A.push_back(a23);
			A.push_back(a33);
			x += 1;
			}
		y += 1;
		}

	//delete [] I;
	delete [] dIdx;
	delete [] dIdy;
	delete [] dIdxdy;
	
	}

void PolarPilatus::getPixelsAtQ( vector<float> & IvsPhi, int q_index,  float q, float a, float b)
{
// populates the vector IvsPhi with interpolated intensity values

  float * coefficients = new float[16];
  for(int phi=0;phi < Nphi; phi++){
	float x = q*cosPhi[phi] + a;
	float y = q*sinPhi[phi] + b;
	int i = int(floor(x));
	int j = int(floor(y));
	int aStart = 16*j*(Xdim-1) + 16*i;
	int k(0);
	while (k < 16){
		coefficients[k] = A[k+aStart];
		k += 1;
		}
	IvsPhi[q_index*Nphi + phi] += EvaluateIntensity( coefficients, x-floor(x), y-floor(y) );
	//IvsPhi[phi] = EvaluateIntensity( coefficients, x-floor(x), y-floor(y) );
	}
  delete [] coefficients;
}

float PolarPilatus::aveNo0(vector<float>& ar)
{
//  does the angular average around a ring of intensities
//  ignores 0 values intensities
  float ave(0);
  float counts(0);
  for(int i=0;i < Nphi; i++){
	if (ar[i] > 0){
		ave += ar[i];
		counts += 1;
		}
	}
  if (counts > 0)
	return ave / counts;
  else
	return 0;
}



void PolarPilatus::Center(float qMin, float qMax, float center_res, int Nphi_, float size)
{ 
/* 
  Finds beam center by BRUTE-FORCING a mximization calculation
  Consider implementing gradient decent if this starts lagging
*/

  Nphi = Nphi_;

  vector<float> maxAA;
  
  float aMin(   float(Xdim)/2. - size  )  ;
  float aMax(   float(Xdim)/2. + size  )  ;
  float bMin(   float(Ydim)/2. - size  )  ;
  float bMax(   float(Ydim)/2. + size  )  ;

  int qdim(0),adim,bdim;
  float q(qMin);
  cout << "\n    Accumulating angular average maxima...\n";
  while (q < qMax){
	//float q_iang = q_res * q + 0.01;
	//float q_pix = tan( 2 * asin( q_iang * wavelen / (4 * M_PI) ) ) * detdist / pixsize ;
	
	float q_pix = q;
	
	bdim=0;
	float b(bMin);
	while (b < bMax){
		adim=0;
		float a(aMin);
		while(a < aMax){
			vector<float> IvsPhi (Nphi,0);
			getPixelsAtQ(IvsPhi, 0, q_pix, a, b);
			float AA = aveNo0( IvsPhi );
			maxAA.push_back( AA );
			a += center_res;
			adim += 1;
			}
		b += center_res;
		bdim += 1;
		}
	q += 1;
	qdim += 1;
	cout << "    --->" << (qMax-qMin - qdim) << endl;
	}

  cout << "    Calculating the center...\n";

  int i(0);
  float max(0);
  int qmax,amax,bmax;
  while( i < qdim-1){
	int j(0);
	while(j < bdim-1){
		int k(0);
		while( k < adim-1){
			if (maxAA[i*bdim*adim + j*adim + k] > max){
				max = maxAA[i*bdim*adim + j*adim + k];
				qmax=i;
				bmax=j;
				amax=k;
				}
			k += 1;
			}
		j += 1;
		}
	i += 1;
	cout << "    ---> " <<  qdim -i << endl;
	}

  qmax = qmax + (int)qMin;
  //float q_iang = q_res * float(qmax) + 0.01;
  //float q_pix = tan( 2 * asin( q_iang * wavelen / (4 * M_PI) ) ) * detdist / pixsize ;

  float q_pix = float(qmax);

  float a_pix = float(amax)*center_res + float(aMin);
  float b_pix = float(bmax)*center_res + float(bMin);

  cout << "\n    The peak intensity value is: " << max << endl;
  cout <<   "    The max parameters are: (q,a,b) (" << q_pix << "," << a_pix << "," << b_pix << ")\n";

  }

PolarPilatus::~PolarPilatus()
{
  cout << "    exiting..." << endl;
}


int main(int argc, char * argv[]){

FILE * binFile = fopen(argv[1],"r");
fseek(binFile,1024,SEEK_SET);
int N = 2463*2527;
float * I = new float[N];
fread(I,4,N,binFile);
fclose(binFile);

PolarPilatus pp(2463,2527,I,0.188,0.000176,0.7293);

pp.Center(300,500,1,100,20);

delete [] I;

return 0;
}

