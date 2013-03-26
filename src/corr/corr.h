
class Corr
{
  int N;
  float ar1_mean;
  float ar2_mean;
  float mean_no_zero(float * ar);
  void correlate(float * ar1, float * ar2, float * ar3);
public:
  Corr(int N_, float * ar1, float * ar2, float * ar3);
  ~Corr();
};

