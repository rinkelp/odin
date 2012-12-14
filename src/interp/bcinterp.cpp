
#include <iostream>
#include <stdexcept>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include "bcinterp.hh"

using namespace std;


Bcinterp::Bcinterp( int Nvals, double *vals, double x_space_, double y_space_,
                    int Xdim_, int Ydim_, double x_corner_, double y_corner_ ) { 
    /*
     * Constructor -- computes the bicubic coefficients `alphas`
     *
     * Compute the bicubic coefficients that definte the interpolation
     * polynomial. These coefficients can then be used in evaluate()
     * to evaluate the interpolated function at any point.
     *
     * Arguments:
     * -- vals: A pointer to an array of function values, z = f(x,y)
     * -- Nvals: The size of the vals array
     * -- x_space, y_space: the pixel spacing in the x/y directions
     * -- Xdim, Ydim: the x/y dimensions of the grid, values on which
     *    are given by `vals`
     */

    // assign members
    x_space = x_space_;
    y_space = y_space_;
    Xdim = Xdim_;
    Ydim = Ydim_;
    x_corner = x_corner_;
    y_corner = y_corner_;
    
    // sanity check
    int N = Xdim * Ydim;
    assert( N == Nvals );

	double * dIdx   = new double[N];
	double * dIdy   = new double[N];
	double * dIdxdy = new double[N];
	
	int x, y;
	double dx, dy, dxdy;
	double a00, a10, a20, a30, a01, a11, a21, a31;
    double a02, a12, a22, a32, a03, a13, a23, a33;
    double * I = vals; // for legacy reasons

    // compute the finite difference derivatives for interior of the grid	
	#pragma omp parallel for shared(dx, dy, dxdy, dIdx, dIdy, dIdxdy) private(x, y)
    for( y = 1; y < Ydim-1; y++ ) {
        for( x = 1; x < Xdim-1; x++ ) {

            // compute finite difference derivatives & store them
			dx = ( F(I,x+1,y) - F(I,x-1,y) ) / 2.0;
			dy = ( F(I,x,y+1) - F(I,x,y-1) ) / 2.0;
			dxdy = ( F(I,x+1,y+1) - F(I,x+1,y-1) - F(I,x-1,y+1) + F(I,x-1,y-1) ) / 4.0;

			dIdx[y*Xdim + x]   = dx;
			dIdy[y*Xdim + x]   = dy;
			dIdxdy[y*Xdim + x] = dxdy;
		}
    }


    #pragma omp parallel for shared(dx, dy, dxdy, dIdx, dIdy, dIdxdy) private(x)
    for( x = 1; x < Xdim-1; x++ ) {

        // top row (no corners)
		dx = dIdx[Xdim + x];
		dy = dIdy[Xdim + x];
		dxdy = dIdxdy[Xdim + x];
		dIdx[x] = dx;
		dIdy[x] = dy;
		dIdxdy[x] = dxdy;

		// do the bottom row (no corners)
		dx= dIdx[Xdim*(Ydim-2) + x];
		dy= dIdy[Xdim*(Ydim-2) + x];
		dxdy= dIdxdy[Xdim*(Ydim-2) + x];
		dIdx[Xdim*(Ydim-1) + x] = dx;
		dIdy[Xdim*(Ydim-1) + x] = dy;
		dIdxdy[Xdim*(Ydim-1) + x] = dxdy;
	}


    #pragma omp parallel for shared(dx, dy, dxdy, dIdx, dIdy, dIdxdy) private(y)
    for( y = 1; y < Ydim-1; y++ ) {

        // do the left side (no corners)
		dx= dIdx[Xdim*y + 1];
		dy= dIdy[Xdim*y + 1];
		dxdy= dIdxdy[Xdim*y + 1];
		dIdx[Xdim*y] = dx;
		dIdy[Xdim*y] = dy;
		dIdxdy[Xdim*y] = dxdy;

		// do the right side (no corners)
		dx= dIdx[Xdim*y + Xdim-2];
		dy= dIdy[Xdim*y + Xdim-2];
		dxdy= dIdxdy[Xdim*y + Xdim-2];
		dIdx[Xdim*y + Xdim-1] = dx;
		dIdy[Xdim*y + Xdim-1] = dy;
		dIdxdy[Xdim*y + Xdim-1] = dxdy;
	}

	// do the top,left
	dx= dIdx[1];
	dy= dIdy[Xdim];
	dxdy = dIdxdy[Xdim+1];
	dIdx[0] = dx;
	dIdy[0] = dy;
	dIdxdy[0] = dxdy;

	// do the top,right
	dx = dIdx[Xdim-2];
	dy = dIdy[2*Xdim-1];
	dxdy = dIdxdy[2*Xdim-2];
	dIdx[Xdim-1] = dx;
	dIdy[Xdim-1] = dy;
	dIdxdy[Xdim-1] = dxdy;

	// do the bottom,left
	dx = dIdx[Xdim*(Ydim-1)+1];
	dy = dIdy[Xdim*(Ydim-2)];
	dxdy = dIdxdy[Xdim*(Ydim-2)+1];
	dIdx[Xdim*(Ydim-1)] = dx;
	dIdy[Xdim*(Ydim-1)] = dy;
	dIdxdy[Xdim*(Ydim-1)] = dxdy;

	// do the bottom,right
	dx = dIdx[Xdim*Ydim-2];
	dy = dIdy[Xdim*(Ydim-1)-1];
	dxdy = dIdxdy[Xdim*(Ydim-1)-2];
	dIdx[Xdim*Ydim-1] = dx;
	dIdy[Xdim*Ydim-1] = dy;
	dIdxdy[Xdim*Ydim-1] = dxdy;


    // compute the the vector alpha by matrix inversion A * \alpha = x
    // for each box on the grid
    size_alphas = (Ydim-1) * (Xdim-1) * 16;
    alphas.resize(size_alphas, 0.0); // generate a vector len aN of zeros

	
    for( x = 0; x < Xdim-1; x++ ) {
        #pragma omp parallel for private(x, y) shared(dx, dy, dxdy, dIdx, dIdy, dIdxdy)
        for( y = 0; y < Ydim-1; y++ ) {

			a00 =    F(I,x,y);
			a10 =    F(dIdx,x,y);
			a20 = -3*F(I,x,y)      + 3*F(I,x+1,y)      - 2*F(dIdx,x,y)    -    F(dIdx,x+1,y);
			a30 =  2*F(I,x,y)      - 2*F(I,x+1,y)      +   F(dIdx,x,y)    +    F(dIdx,x+1,y);

			a01 =    F(dIdy,x,y);
			a11 =    F(dIdxdy,x,y);
			a21 = -3*F(dIdy,x,y)   + 3*F(dIdy,x+1,y)   - 2*F(dIdxdy,x,y)   -   F(dIdxdy,x+1,y);
			a31 =  2*F(dIdy,x,y)   - 2*F(dIdy,x+1,y)   +   F(dIdxdy,x,y)   +   F(dIdxdy,x+1,y);

			a02 = -3*F(I,x,y)      + 3*F(I,x,y+1)      - 2*F(dIdy,x,y)     -   F(dIdy,x,y+1);
			a12 = -3*F(dIdx,x,y)   + 3*F(dIdx,x,y+1)   - 2*F(dIdxdy,x,y)   -   F(dIdxdy,x,y+1);
			a22 =  9*F(I,x,y)      - 9*F(I,x+1,y)      - 9*F(I,x,y+1)      + 9*F(I,x+1,y+1);
			a22+=  6*F(dIdx,x,y)   + 3*F(dIdx,x+1,y)   - 6*F(dIdx,x,y+1)   - 3*F(dIdx,x+1,y+1);
			a22+=  6*F(dIdy,x,y)   - 6*F(dIdy,x+1,y)   + 3*F(dIdy,x,y+1)   - 3*F(dIdy,x+1,y+1);
			a22+=  4*F(dIdxdy,x,y) + 2*F(dIdxdy,x+1,y) + 2*F(dIdxdy,x,y+1) +   F(dIdxdy,x+1,y+1);
			a32 = -6*F(I,x,y)      + 6*F(I,x+1,y)      + 6*F(I,x,y+1)      - 6*F(I,x+1,y+1);
			a32+= -3*F(dIdx,x,y)   - 3*F(dIdx,x+1,y)   + 3*F(dIdx,x,y+1)   + 3*F(dIdx,x+1,y+1);
			a32+= -4*F(dIdy,x,y)   + 4*F(dIdy,x+1,y)   - 2*F(dIdy,x,y+1)   + 2*F(dIdy,x+1,y+1);
			a32+= -2*F(dIdxdy,x,y) - 2*F(dIdxdy,x+1,y) -   F(dIdxdy,x,y+1) -   F(dIdxdy,x+1,y+1);

			a03 =  2*F(I,x,y)      - 2*F(I,x,y+1)      +   F(dIdy,x,y)     +   F(dIdy,x,y+1);
			a13 =  2*F(dIdx,x,y)   - 2*F(dIdx,x,y+1)   +   F(dIdxdy,x,y)   +   F(dIdxdy,x,y+1);
			a23 = -6*F(I,x,y)      + 6*F(I,x+1,y)      + 6*F(I,x,y+1)      - 6*F(I,x+1,y+1);
			a23+= -4*F(dIdx,x,y)   - 2*F(dIdx,x+1,y)   + 4*F(dIdx,x,y+1)   + 2*F(dIdx,x+1,y+1);
			a23+= -3*F(dIdy,x,y)   + 3*F(dIdy,x+1,y)   - 3*F(dIdy,x,y+1)   + 3*F(dIdy,x+1,y+1);
			a23+= -2*F(dIdxdy,x,y) -   F(dIdxdy,x+1,y) - 2*F(dIdxdy,x,y+1) -   F(dIdxdy,x+1,y+1);
		    a33 =  4*F(I,x,y)      - 4*F(I,x+1,y)      - 4*F(I,x,y+1)      + 4*F(I,x+1,y+1);
			a33+=  2*F(dIdx,x,y)   + 2*F(dIdx,x+1,y)   - 2*F(dIdx,x,y+1)   - 2*F(dIdx,x+1,y+1);
			a33+=  2*F(dIdy,x,y)   - 2*F(dIdy,x+1,y)   + 2*F(dIdy,x,y+1)   - 2*F(dIdy,x+1,y+1);
			a33+=    F(dIdxdy,x,y) +   F(dIdxdy,x+1,y) +   F(dIdxdy,x,y+1) +   F(dIdxdy,x+1,y+1);

            // store the computed values
            unsigned int k = 16*x*(Ydim-1) + 16*y;
            
            if( (k + 15) >= size_alphas ) {
                cout << "accessing pos: " << k+15 << endl;
                cout << "over-run range on alphas in Bcinterp constructor" << endl;
                throw std::out_of_range("over-run range on alphas in Bcinterp constructor");
            }

            alphas[k] = a00;
            alphas[k+1] = a10;
            alphas[k+2] = a20;
            alphas[k+3] = a30;
            alphas[k+4] = a01;
            alphas[k+5] = a11;
            alphas[k+6] = a21;
            alphas[k+7] = a31;
            alphas[k+8] = a02;
            alphas[k+9] = a12;
            alphas[k+10] = a22;
            alphas[k+11] = a32;
            alphas[k+12] = a03;
            alphas[k+13] = a13;
            alphas[k+14] = a23;
            alphas[k+15] = a33;
	    }
	}

	delete [] dIdx;
	delete [] dIdy;
	delete [] dIdxdy;
}


double Bcinterp::F (double vals[], int i, int j) {
    // a helper function to evaluate the function F at index (i,j)
	double val = vals[j*Xdim+i];
	return val;
}


double Bcinterp::evaluate_point (double x, double y) {
    /* 
     * Evaluate the interpolation defined by alpha at point (x,y)
     */

    double interpI;
    double a00, a10, a20, a30, a01, a11, a21, a31;
    double a02, a12, a22, a32, a03, a13, a23, a33;

    // map the point (x,y) to the indicies of our interpolated grid
    double xm = (x-x_corner) / x_space;
    double ym = (y-y_corner) / y_space;
    
    if( xm < 0.0 ) {
        cout << "xm less than zero" << endl;
        throw std::out_of_range("xm less than zero");
    }
    if( ym < 0.0 ) {
        cout << "ym less than zero" << endl;
        throw std::out_of_range("ym less than zero");
    }

    // choose which square we're in by looking at the top-right
    unsigned int i = floor(xm);
	unsigned int j = floor(ym);

	// retrieve the alpha parameters for that specific square (these define
	// the polynomial function over the square)
	unsigned int aStart = 16*i*(Ydim-1) + 16*j;
	
    if( (aStart + 15) >= size_alphas ) {
        cout << "accessing pos: " << aStart+15 << endl;
        cout << "over-run range on alphas in evaluate_point" << endl;
        throw std::out_of_range("over-run range on alphas in evaluate_point");
    }
	
    a00 = alphas[aStart + 0];
    a10 = alphas[aStart + 1];
    a20 = alphas[aStart + 2];
    a30 = alphas[aStart + 3];
    a01 = alphas[aStart + 4];
    a11 = alphas[aStart + 5];
    a21 = alphas[aStart + 6];
    a31 = alphas[aStart + 7];
    a02 = alphas[aStart + 8];
    a12 = alphas[aStart + 9];
    a22 = alphas[aStart + 10];
    a32 = alphas[aStart + 11];
    a03 = alphas[aStart + 12];
    a13 = alphas[aStart + 13];
    a23 = alphas[aStart + 14];
    a33 = alphas[aStart + 15];

    // evaluate the point on the square
    double xp = xm - (double) i;
    double yp = ym - (double) j;
    
	interpI =  a00          + a10*xp          + a20*xp*xp          + a30*xp*xp*xp;
    interpI += a01*yp       + a11*xp*yp       + a21*xp*xp*yp       + a31*xp*xp*xp*yp;
    interpI += a02*yp*yp    + a12*xp*yp*yp    + a22*xp*xp*yp*yp    + a32*xp*xp*xp*yp*yp;
    interpI += a03*yp*yp*yp + a13*xp*yp*yp*yp + a23*xp*xp*yp*yp*yp + a33*xp*xp*xp*yp*yp*yp;

    return interpI;
}


void Bcinterp::evaluate_array(int dim_xa, double *xa, int dim_ya, double *ya, 
                              int dim_za, double *za) {
                                  
    // evaluate an array of points f(x,y) with x/y vectors
    // here, za is the output array, dim_xa must == dim_ya
    if( dim_xa != dim_ya || dim_xa != dim_za ) {
        cout << "xa, ya, za must all be same dimension" << endl;
        throw std::invalid_argument("xa, ya, za must all be same dimension");
    }
        
    #pragma omp parallel for shared(za)
    for( int i = 0; i < dim_za; i++ ) {
        za[i] = evaluate_point(xa[i], ya[i]);
    }
    
}


Bcinterp::~Bcinterp() {
    //delete [] alphas;
}


