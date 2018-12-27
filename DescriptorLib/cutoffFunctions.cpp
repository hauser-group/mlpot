#include "cutoffFunctions.h"
#include <stdio.h>
#include <math.h>

CutoffFunction::CutoffFunction(double cutoff)
{
    this->cutoff = cutoff;
};

CutoffFunction::CutoffFunction(){};

CutoffFunction::~CutoffFunction(){};

double CutoffFunction::eval(double r)
{
  return 0.0;
};

double ConstCutoffFunction::eval(double r)
{
  if (r <= cutoff) return 1.0;
  else return 0.0;
};

double ConstCutoffFunction::derivative(double r)
{
    return 0.0;
};

double CosCutoffFunction::eval(double r)
{
    if (r <= cutoff) return 0.5*(cos(M_PI*r/cutoff)+1.0);
    else return 0.0;
};

double CosCutoffFunction::derivative(double r)
{
    if (r <= cutoff) return -0.5*(M_PI*sin(M_PI*r/cutoff))/cutoff;
    else return 0.0;
};

double TanhCutoffFunction::eval(double r)
{
    if (r <= cutoff) return pow(tanh(1.0-r/cutoff),3);
    else return 0.0;
};

double TanhCutoffFunction::derivative(double r)
{
    if (r <= cutoff) return -(3.0*pow(sinh(1.0-r/cutoff),2))/
        (cutoff*pow(cosh(1.0-r/cutoff),4));
    else return 0.0;
};

double PolynomialCutoffFunction::eval(double r)
{
    if (r <= cutoff)
    {
      return 1.0 - 10.0 * pow(r/cutoff, 3) + 15.0 * pow(r/cutoff, 4) -
        6.0 * pow(r/cutoff, 5);
    }
    else return 0.0;
};

double PolynomialCutoffFunction::derivative(double r)
{
  if (r <= cutoff)
  {
    return -30.0 * pow(r/cutoff, 2)/cutoff + 60.0 * pow(r/cutoff, 3)/cutoff -
      30.0 * pow(r/cutoff, 4)/cutoff;
  }
  else return 0.0;
};

// From https://en.wikipedia.org/wiki/Non-analytic_smooth_function
double SmoothCutoffFunction::eval(double r)
{
    if (r <= cutoff)
    {
      auto x0 = exp(-cutoff/r);
      return 1.0 - x0/(x0+exp(-cutoff/(cutoff-r)));
    }
    else return 0.0;
};

double SmoothCutoffFunction::derivative(double r)
{
    if (r <= cutoff)
    {
      auto x0 = exp(-cutoff/r);
      auto x1 = exp(-cutoff/(cutoff-r));
      return (cutoff*x0)/(pow(r,2)*(x0+x1))-(x0*((cutoff*x0)/(r*r)-
        (cutoff*x1)/pow(cutoff-r,2)))/pow(x0+x1,2);
    }
    else return 0.0;
};

double Smooth2CutoffFunction::eval(double r)
{
    if (r < cutoff)
    {
      return exp(1.0 - 1.0/(1.0-pow(r/cutoff,2)));
    }
    else return 0.0;
};

double Smooth2CutoffFunction::derivative(double r)
{
    if (r < cutoff)
    {
      auto x0 = pow(r,2)-pow(cutoff,2);
      return -(2.0*pow(cutoff,2)*r*exp(pow(r,2)/x0))/pow(x0,2);
    }
    else return 0.0;
};

double ShortRangeCutoffFunction::eval(double r)
{
    return 1.0 / (1.0 + exp(10.0*(r-cutoff)));
};

double ShortRangeCutoffFunction::derivative(double r)
{
    return -10.0*exp(10.0*(r-cutoff))/pow(exp(10.0*(r-cutoff)) + 1.0, 2);
};

double LongRangeCutoffFunction::eval(double r)
{
    return 1.0 / (1.0 + exp(-10.0*(r-cutoff)));
};

double LongRangeCutoffFunction::derivative(double r)
{
    return 10.0*exp(-10.0*(r-cutoff))/pow(exp(-10.0*(r-cutoff)) + 1.0, 2);
};
