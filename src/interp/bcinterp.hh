
#include <vector>

class Bcinterp
{
    
public:    
    Bcinterp(int Nvals, double *vals, double x_space_, double y_space_, int Xdim_, 
             int Ydim_, double x_corner_, double y_corner_);
    double evaluate_point(double x, double y);
    void evaluate_array(int dim_xa, double *xa, int dim_ya, double *ya, 
                        int dim_za, double *za);
                        
    double x_space, y_space, x_corner, y_corner;
    int Xdim, Ydim;

    ~Bcinterp();
    
private:
    std::vector<double> alphas;
    unsigned int size_alphas;
    double F(double vals[], int i, int j);
};