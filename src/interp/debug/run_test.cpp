#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include "bcinterp.cpp"

using namespace std;

int main(){

    cout << "running test...\n";

    int Xdim = 1000;
    int Ydim = 1000;
    int Nvals = Xdim * Ydim;
        
    float * vals = new float [Nvals];
    for(int i = 0; i < Nvals; i++) {
        vals[i] = float(i);
    }
    
    float x_space = 0.1;
    float y_space = 0.1;

    Bcinterp bc ( Nvals, vals, x_space,  y_space, Xdim,  Ydim, 0.0, 0.0 );
          
    float i, x, y;
    x = 1.01;
    y = 1.01;
    i = bc.evaluate_point(x, y);
    cout << "interp value: " << i << endl;

    delete [] vals;

    return 0;
}
