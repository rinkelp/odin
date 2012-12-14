#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include "bcinterp.cpp"

using namespace std;

int main(){

    cout << "running test...\n";

    int Xdim = 10000;
    int Ydim = 10000;
    int Nvals = Xdim * Ydim;
        
    double * vals = new double [Nvals];
    for(int i = 0; i < Nvals; i++) {
        vals[i] = float(i);
    }
    
    double x_space = 0.1;
    double y_space = 0.1;

    Bcinterp bc ( Nvals, vals, x_space,  y_space, Xdim,  Ydim, 0.0, 0.0 );
          
    double i, x, y;
    x = 1.01;
    y = 1.01;
    i = bc.evaluate_point(x, y);
    cout << "interp value: " << i << endl;

    delete [] vals;

    return 0;
}
