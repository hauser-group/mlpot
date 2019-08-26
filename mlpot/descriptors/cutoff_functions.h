class CutoffFunction
{
    public:
      CutoffFunction(double cutoff);
      CutoffFunction();
      virtual ~CutoffFunction();
      virtual double eval(double r) = 0;
      virtual double derivative(double r) = 0;
    protected:
      double cutoff;
};

class ConstCutoffFunction: public CutoffFunction
{
    public:
        ConstCutoffFunction(double cutoff):CutoffFunction(cutoff){};
        double eval(double r);
        double derivative(double r);
};

class CosCutoffFunction: public CutoffFunction
{
    public:
        CosCutoffFunction(double cutoff):CutoffFunction(cutoff){};
        double eval(double r);
        double derivative(double r);
};

class TanhCutoffFunction: public CutoffFunction
{
public:
  TanhCutoffFunction(double cutoff):CutoffFunction(cutoff){};
  double eval(double r);
  double derivative(double r);
};

class PolynomialCutoffFunction: public CutoffFunction
{
public:
  PolynomialCutoffFunction(double cutoff):CutoffFunction(cutoff){};
  double eval(double r);
  double derivative(double r);
};

class SmoothCutoffFunction: public CutoffFunction
{
public:
  SmoothCutoffFunction(double cutoff):CutoffFunction(cutoff){};
  double eval(double r);
  double derivative(double r);
};

class Smooth2CutoffFunction: public CutoffFunction
{
public:
  Smooth2CutoffFunction(double cutoff):CutoffFunction(cutoff){};
  double eval(double r);
  double derivative(double r);
};

class ShortRangeCutoffFunction: public CutoffFunction
{
public:
  ShortRangeCutoffFunction(double cutoff):CutoffFunction(cutoff){};
  double eval(double r);
  double derivative(double r);
};

class LongRangeCutoffFunction: public CutoffFunction
{
public:
  LongRangeCutoffFunction(double cutoff):CutoffFunction(cutoff){};
  double eval(double r);
  double derivative(double r);
};
