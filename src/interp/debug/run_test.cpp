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
    cout << "(1.01, 1.01): " << i << endl;

    x = 1.5;
    y = 1.5;
    i = bc.evaluate_point(x, y);
    cout << "(1.5, 1.5): " << i << endl;

    x = 20.3;
    y = 2.0;
    i = bc.evaluate_point(x, y);
    cout << "(20.3, 2.0): " << i << endl;


    delete [] vals;

    return 0;
}
