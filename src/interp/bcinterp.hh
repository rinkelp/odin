
#include <vector>

class Bcinterp
{
    
public:    
    Bcinterp(int Nvals, float *vals, float x_space_, float y_space_, int Xdim_, 
             int Ydim_, float x_corner_, float y_corner_);
    float evaluate_point(float x, float y);
    void evaluate_array(int dim_xa, float *xa, int dim_ya, float *ya, 
                        int dim_za, float *za);
                        
    float x_space, y_space, x_corner, y_corner;
    int Xdim, Ydim;

    ~Bcinterp();
    
private:
    std::vector<float> alphas;
    unsigned int size_alphas;
    float F(float vals[], int i, int j);
};

