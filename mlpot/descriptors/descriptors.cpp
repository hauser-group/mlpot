/*

CAUTION: Part of this file is written by the python script generateSymFuns.py.
These parts are marked by comment lines starting with AUTOMATIC. Do not alter
anything between these tags.
*/

#include "descriptors.h"
#include <stdio.h>
#include <math.h>
#include <limits>
#include <string.h>

Descriptor::Descriptor(int num_prms, double* prms,
  std::shared_ptr<CutoffFunction> cutfun):cutfun(cutfun)
{
  this->num_prms = num_prms;
  this->prms = new double[num_prms];
  for (int i = 0; i < num_prms; i++)
  {
    this->prms[i] = prms[i];
  }
};

Descriptor::~Descriptor()
{
  delete[] prms;
};

Descriptor::Descriptor(const Descriptor& other) //Copy constructor
{
  num_prms = other.num_prms;
  prms = new double[other.num_prms];
  for (int i = 0; i < num_prms; i++)
  {
    prms[i] = other.prms[i];
  }
};

Descriptor& Descriptor::operator=(const Descriptor& other) //Copy assignment
{
  double* tmp_prms = new double[other.num_prms];
  for (int i = 0; i < other.num_prms; i++)
  {
    tmp_prms[i] = other.prms[i];
  }
  delete[] prms;
  prms = tmp_prms;
  return *this;
};

double BehlerG1::eval(double rij)
{
  return cutfun->eval(rij);
};

double BehlerG1::drij(double rij)
{
  return cutfun->derivative(rij);
};

void BehlerG1::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  G = cutfun->eval(rij);
  dGdrij = cutfun->derivative(rij);
};

double BehlerG2::eval(double rij)
{
  return cutfun->eval(rij)*exp(-prms[0]*pow(-prms[1] + rij, 2));
};

double BehlerG2::drij(double rij)
{
  return (2*prms[0]*(prms[1] - rij)*cutfun->eval(rij) + cutfun->derivative(rij))*exp(-prms[0]*pow(prms[1] - rij, 2));
};

void BehlerG2::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  auto x0 = cutfun->eval(rij);
  auto x1 = prms[1] - rij;
  auto x2 = exp(-prms[0]*pow(x1, 2));
  G = x0*x2;
  dGdrij = x2*(2*prms[0]*x0*x1 + cutfun->derivative(rij));
};

double BehlerG3::eval(double rij)
{
  return cutfun->eval(rij)*cos(prms[0]*rij);
};

double BehlerG3::drij(double rij)
{
  return -prms[0]*cutfun->eval(rij)*sin(prms[0]*rij) + cutfun->derivative(rij)*cos(prms[0]*rij);
};

void BehlerG3::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  auto x0 = cutfun->eval(rij);
  auto x1 = prms[0]*rij;
  auto x2 = cos(x1);
  G = x0*x2;
  dGdrij = -prms[0]*x0*sin(x1) + x2*cutfun->derivative(rij);
};

// AUTOMATIC custom TwoBodyDescriptors start

double BehlerG1old::eval(double rij)
{
  return cutfun->eval(rij)*exp(-pow(rij, 2)*prms[0]);
};

double BehlerG1old::drij(double rij)
{
  return (-2*rij*cutfun->eval(rij)*prms[0] + cutfun->derivative(rij))*exp(-pow(rij, 2)*prms[0]);
};

void BehlerG1old::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  auto x0 = cutfun->eval(rij);
  auto x1 = exp(-pow(rij, 2)*prms[0]);
  G = x0*x1;
  dGdrij = x1*(-2*rij*x0*prms[0] + cutfun->derivative(rij));
};

double OneOverR6::eval(double rij)
{
  return cutfun->eval(rij)/pow(rij, 6);
};

double OneOverR6::drij(double rij)
{
  return (rij*cutfun->derivative(rij) - 6*cutfun->eval(rij))/pow(rij, 7);
};

void OneOverR6::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  auto x0 = cutfun->eval(rij);
  G = x0/pow(rij, 6);
  dGdrij = (rij*cutfun->derivative(rij) - 6*x0)/pow(rij, 7);
};

double OneOverR8::eval(double rij)
{
  return cutfun->eval(rij)/pow(rij, 8);
};

double OneOverR8::drij(double rij)
{
  return (rij*cutfun->derivative(rij) - 8*cutfun->eval(rij))/pow(rij, 9);
};

void OneOverR8::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  auto x0 = cutfun->eval(rij);
  G = x0/pow(rij, 8);
  dGdrij = (rij*cutfun->derivative(rij) - 8*x0)/pow(rij, 9);
};

double OneOverR10::eval(double rij)
{
  return cutfun->eval(rij)/pow(rij, 10);
};

double OneOverR10::drij(double rij)
{
  return (rij*cutfun->derivative(rij) - 10*cutfun->eval(rij))/pow(rij, 11);
};

void OneOverR10::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  auto x0 = cutfun->eval(rij);
  G = x0/pow(rij, 10);
  dGdrij = (rij*cutfun->derivative(rij) - 10*x0)/pow(rij, 11);
};

double radialTest::eval(double rij)
{
  return rij;
};

double radialTest::drij(double rij)
{
  return 1;
};

void radialTest::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  G = rij;
  dGdrij = 1;
};
// AUTOMATIC custom TwoBodyDescriptors end

double BehlerG4::eval(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*exp2(1 - prms[1])*cutfun->eval(rij)*cutfun->eval(rik)*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*exp(-2*(-costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*prms[2]);
};

double BehlerG4::drij(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-(costheta*rik - rij)*cutfun->derivative(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*cutfun->eval(rij) + (2*(costheta*rik - 2*rij)*cutfun->eval(rij)*prms[2] + cutfun->derivative(rij))*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))))*exp2(1 - prms[1])*cutfun->eval(rik)*exp(-2*(-costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*prms[2])/sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2));
};

double BehlerG4::drik(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-(costheta*rij - rik)*cutfun->derivative(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*cutfun->eval(rik) + (2*(costheta*rij - 2*rik)*cutfun->eval(rik)*prms[2] + cutfun->derivative(rik))*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))))*exp2(1 - prms[1])*cutfun->eval(rij)*exp(-2*(-costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*prms[2])/sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2));
};

double BehlerG4::dcostheta(double rij, double rik, double costheta)
{
  return (2*rij*rik*pow(costheta*prms[0] + 1, prms[1] + 1)*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*prms[2] - rij*rik*pow(costheta*prms[0] + 1, prms[1] + 1)*cutfun->derivative(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))) + pow(costheta*prms[0] + 1, prms[1])*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*prms[0]*prms[1])*exp2(1 - prms[1])*cutfun->eval(rij)*cutfun->eval(rik)*exp(-2*(-costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*prms[2])/((costheta*prms[0] + 1 + std::numeric_limits<double>::epsilon())*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)));
};

void BehlerG4::eval_with_derivatives(double rij, double rik, double costheta,
  double &G, double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = 2*rij;
  auto x1 = costheta*rik;
  auto x2 = pow(rij, 2) + pow(rik, 2);
  auto x3 = sqrt(-x0*x1 + x2);
  auto x4 = cutfun->eval(x3);
  auto x5 = cutfun->eval(rij);
  auto x6 = cutfun->eval(rik);
  auto x7 = costheta*prms[0] + 1;
  auto x8 = pow(x7, prms[1]);
  auto x9 = 2*prms[2];
  auto x10 = exp(-x9*(-rij*x1 + x2));
  auto x11 = exp2(1 - prms[1]);
  auto x12 = x10*x11*x6*x8;
  auto x13 = 1.0/x3;
  auto x14 = cutfun->derivative(x3);
  auto x15 = x3*x4;
  auto x16 = costheta*rij;
  auto x17 = x10*x11*x13*x5;
  auto x18 = rik*pow(x7, prms[1] + 1);
  G = x12*x4*x5;
  dGdrij = x12*x13*(-x14*x5*(-rij + x1) + x15*(x5*x9*(-x0 + x1) + cutfun->derivative(rij)));
  dGdrik = x17*x8*(-x14*x6*(-rik + x16) + x15*(x6*x9*(-2*rik + x16) + cutfun->derivative(rik)));
  dGdcostheta = x17*x6*(-rij*x14*x18 + x0*x15*x18*prms[2] + x15*x8*prms[0]*prms[1])/(x7 + std::numeric_limits<double>::epsilon());
};

void BehlerG4::derivatives(double rij, double rik, double costheta,
  double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = cutfun->eval(rik);
  auto x1 = cutfun->eval(rij);
  auto x2 = costheta*rik;
  auto x3 = 2*rij;
  auto x4 = pow(rij, 2) + pow(rik, 2);
  auto x5 = sqrt(-x2*x3 + x4);
  auto x6 = cutfun->derivative(x5);
  auto x7 = 2*prms[2];
  auto x8 = x5*cutfun->eval(x5);
  auto x9 = costheta*prms[0] + 1;
  auto x10 = pow(x9, prms[1]);
  auto x11 = 1.0/x5;
  auto x12 = exp(-x7*(-rij*x2 + x4));
  auto x13 = exp2(1 - prms[1]);
  auto x14 = x10*x11*x12*x13;
  auto x15 = costheta*rij;
  auto x16 = rik*pow(x9, prms[1] + 1);
  dGdrij = x0*x14*(-x1*x6*(-rij + x2) + x8*(x1*x7*(x2 - x3) + cutfun->derivative(rij)));
  dGdrik = x1*x14*(-x0*x6*(-rik + x15) + x8*(x0*x7*(-2*rik + x15) + cutfun->derivative(rik)));
  dGdcostheta = x0*x1*x11*x12*x13*(-rij*x16*x6 + x10*x8*prms[0]*prms[1] + x16*x3*x8*prms[2])/(x9 + std::numeric_limits<double>::epsilon());
};

double BehlerG5::eval(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*exp2(-prms[1] + 1)*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG5::drij(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-2*prms[2]*rij*cutfun->eval(rij) + cutfun->derivative(rij))*exp2(-prms[1] + 1)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG5::drik(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-2*prms[2]*rik*cutfun->eval(rik) + cutfun->derivative(rik))*exp2(-prms[1] + 1)*cutfun->eval(rij)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG5::dcostheta(double rij, double rik, double costheta)
{
  return prms[0]*prms[1]*pow(costheta*prms[0] + 1, prms[1] - 1)*exp2(-prms[1] + 1)*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

void BehlerG5::eval_with_derivatives(double rij, double rik, double costheta,
  double &G, double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = costheta*prms[0] + 1;
  auto x1 = pow(x0, prms[1]);
  auto x2 = cutfun->eval(rij);
  auto x3 = cutfun->eval(rik);
  auto x4 = exp2(-prms[1] + 1);
  auto x5 = exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
  auto x6 = x2*x3*x4*x5;
  auto x7 = 2*prms[2];
  auto x8 = x1*x4*x5;
  G = x1*x6;
  dGdrij = x3*x8*(-rij*x2*x7 + cutfun->derivative(rij));
  dGdrik = x2*x8*(-rik*x3*x7 + cutfun->derivative(rik));
  dGdcostheta = prms[0]*prms[1]*pow(x0, prms[1] - 1)*x6;
};

void BehlerG5::derivatives(double rij, double rik, double costheta,
  double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = costheta*prms[0] + 1;
  auto x1 = pow(x0, prms[1]);
  auto x2 = 2*prms[2];
  auto x3 = cutfun->eval(rij);
  auto x4 = cutfun->eval(rik);
  auto x5 = exp2(-prms[1] + 1);
  auto x6 = exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
  auto x7 = x4*x5*x6;
  dGdrij = x1*x7*(-rij*x2*x3 + cutfun->derivative(rij));
  dGdrik = x1*x3*x5*x6*(-rik*x2*x4 + cutfun->derivative(rik));
  dGdcostheta = prms[0]*prms[1]*pow(x0, prms[1] - 1)*x3*x7;
};

// AUTOMATIC custom ThreeBodyDescriptors start

double BehlerG4old::eval(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*exp2(1 - prms[1])*cutfun->eval(rij)*cutfun->eval(rik)*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*exp(-2*(-costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*prms[2]);
};

double BehlerG4old::drij(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-(costheta*rik - rij)*cutfun->derivative(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*cutfun->eval(rij) + (2*(costheta*rik - 2*rij)*cutfun->eval(rij)*prms[2] + cutfun->derivative(rij))*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))))*exp2(1 - prms[1])*cutfun->eval(rik)*exp(-2*(-costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*prms[2])/sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2));
};

double BehlerG4old::drik(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-(costheta*rij - rik)*cutfun->derivative(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*cutfun->eval(rik) + (2*(costheta*rij - 2*rik)*cutfun->eval(rik)*prms[2] + cutfun->derivative(rik))*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))))*exp2(1 - prms[1])*cutfun->eval(rij)*exp(-2*(-costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*prms[2])/sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2));
};

double BehlerG4old::dcostheta(double rij, double rik, double costheta)
{
  return (2*rij*rik*pow(costheta*prms[0] + 1, prms[1] + 1)*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*prms[2] - rij*rik*pow(costheta*prms[0] + 1, prms[1] + 1)*cutfun->derivative(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))) + pow(costheta*prms[0] + 1, prms[1])*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*prms[0]*prms[1])*exp2(1 - prms[1])*cutfun->eval(rij)*cutfun->eval(rik)*exp(-2*(-costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*prms[2])/((costheta*prms[0] + 1)*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)));
};

void BehlerG4old::eval_with_derivatives(double rij, double rik, double costheta,
  double &G, double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = 2*rij;
  auto x1 = costheta*rik;
  auto x2 = pow(rij, 2) + pow(rik, 2);
  auto x3 = sqrt(-x0*x1 + x2);
  auto x4 = cutfun->eval(x3);
  auto x5 = cutfun->eval(rij);
  auto x6 = cutfun->eval(rik);
  auto x7 = costheta*prms[0] + 1;
  auto x8 = pow(x7, prms[1]);
  auto x9 = 2*prms[2];
  auto x10 = exp(-x9*(-rij*x1 + x2));
  auto x11 = exp2(1 - prms[1]);
  auto x12 = x10*x11*x6*x8;
  auto x13 = 1.0/x3;
  auto x14 = cutfun->derivative(x3);
  auto x15 = x3*x4;
  auto x16 = costheta*rij;
  auto x17 = x10*x11*x13*x5;
  auto x18 = rik*pow(x7, prms[1] + 1);
  G = x12*x4*x5;
  dGdrij = x12*x13*(-x14*x5*(-rij + x1) + x15*(x5*x9*(-x0 + x1) + cutfun->derivative(rij)));
  dGdrik = x17*x8*(-x14*x6*(-rik + x16) + x15*(x6*x9*(-2*rik + x16) + cutfun->derivative(rik)));
  dGdcostheta = x17*x6*(-rij*x14*x18 + x0*x15*x18*prms[2] + x15*x8*prms[0]*prms[1])/x7;
};

void BehlerG4old::derivatives(double rij, double rik, double costheta,
  double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = cutfun->eval(rik);
  auto x1 = cutfun->eval(rij);
  auto x2 = costheta*rik;
  auto x3 = 2*rij;
  auto x4 = pow(rij, 2) + pow(rik, 2);
  auto x5 = sqrt(-x2*x3 + x4);
  auto x6 = cutfun->derivative(x5);
  auto x7 = 2*prms[2];
  auto x8 = x5*cutfun->eval(x5);
  auto x9 = costheta*prms[0] + 1;
  auto x10 = pow(x9, prms[1]);
  auto x11 = 1.0/x5;
  auto x12 = exp(-x7*(-rij*x2 + x4));
  auto x13 = exp2(1 - prms[1]);
  auto x14 = x10*x11*x12*x13;
  auto x15 = costheta*rij;
  auto x16 = rik*pow(x9, prms[1] + 1);
  dGdrij = x0*x14*(-x1*x6*(-rij + x2) + x8*(x1*x7*(x2 - x3) + cutfun->derivative(rij)));
  dGdrik = x1*x14*(-x0*x6*(-rik + x15) + x8*(x0*x7*(-2*rik + x15) + cutfun->derivative(rik)));
  dGdcostheta = x0*x1*x11*x12*x13*(-rij*x16*x6 + x10*x8*prms[0]*prms[1] + x16*x3*x8*prms[2])/x9;
};

double BehlerG5mod::eval(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*exp2(1 - prms[1])*cutfun->eval(rij)*cutfun->eval(rik)*exp(-(pow(rij - prms[3], 2) + pow(rik - prms[4], 2))*prms[2]);
};

double BehlerG5mod::drij(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-2*(rij - prms[3])*cutfun->eval(rij)*prms[2] + cutfun->derivative(rij))*exp2(1 - prms[1])*cutfun->eval(rik)*exp(-(pow(rij - prms[3], 2) + pow(rik - prms[4], 2))*prms[2]);
};

double BehlerG5mod::drik(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-2*(rik - prms[4])*cutfun->eval(rik)*prms[2] + cutfun->derivative(rik))*exp2(1 - prms[1])*cutfun->eval(rij)*exp(-(pow(rij - prms[3], 2) + pow(rik - prms[4], 2))*prms[2]);
};

double BehlerG5mod::dcostheta(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1] - 1)*exp2(1 - prms[1])*cutfun->eval(rij)*cutfun->eval(rik)*exp(-(pow(rij - prms[3], 2) + pow(rik - prms[4], 2))*prms[2])*prms[0]*prms[1];
};

void BehlerG5mod::eval_with_derivatives(double rij, double rik, double costheta,
  double &G, double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = cutfun->eval(rij);
  auto x1 = cutfun->eval(rik);
  auto x2 = costheta*prms[0] + 1;
  auto x3 = pow(x2, prms[1]);
  auto x4 = rij - prms[3];
  auto x5 = rik - prms[4];
  auto x6 = exp(-(pow(x4, 2) + pow(x5, 2))*prms[2]);
  auto x7 = exp2(1 - prms[1]);
  auto x8 = x1*x3*x6*x7;
  auto x9 = 2*prms[2];
  auto x10 = x0*x6*x7;
  G = x0*x8;
  dGdrij = x8*(-x0*x4*x9 + cutfun->derivative(rij));
  dGdrik = x10*x3*(-x1*x5*x9 + cutfun->derivative(rik));
  dGdcostheta = x1*x10*pow(x2, prms[1] - 1)*prms[0]*prms[1];
};

void BehlerG5mod::derivatives(double rij, double rik, double costheta,
  double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = cutfun->eval(rik);
  auto x1 = rij - prms[3];
  auto x2 = cutfun->eval(rij);
  auto x3 = 2*prms[2];
  auto x4 = costheta*prms[0] + 1;
  auto x5 = rik - prms[4];
  auto x6 = exp(-(pow(x1, 2) + pow(x5, 2))*prms[2]);
  auto x7 = exp2(1 - prms[1]);
  auto x8 = pow(x4, prms[1])*x6*x7;
  dGdrij = x0*x8*(-x1*x2*x3 + cutfun->derivative(rij));
  dGdrik = x2*x8*(-x0*x3*x5 + cutfun->derivative(rik));
  dGdcostheta = x0*x2*pow(x4, prms[1] - 1)*x6*x7*prms[0]*prms[1];
};
// AUTOMATIC custom ThreeBodyDescriptors end

std::shared_ptr<CutoffFunction> switch_cutoff_functions(
  int cutoff_type, double cutoff)
{
  std::shared_ptr<CutoffFunction> cutfun;
  if (cutoff_type == 0) cutfun = std::make_shared<ConstCutoffFunction>(cutoff);
  else if (cutoff_type == 1) cutfun = std::make_shared<CosCutoffFunction>(cutoff);
  else if (cutoff_type == 2) cutfun = std::make_shared<TanhCutoffFunction>(cutoff);
  else if (cutoff_type == 3) cutfun = std::make_shared<PolynomialCutoffFunction>(cutoff);
  else if (cutoff_type == 4) cutfun = std::make_shared<SmoothCutoffFunction>(cutoff);
  else if (cutoff_type == 5) cutfun = std::make_shared<Smooth2CutoffFunction>(cutoff);
  else if (cutoff_type == 6) cutfun = std::make_shared<ShortRangeCutoffFunction>(cutoff);
  else if (cutoff_type == 7) cutfun = std::make_shared<LongRangeCutoffFunction>(cutoff);
  return cutfun;
}

std::shared_ptr<TwoBodyDescriptor> switch_two_body_descriptors(
  int funtype, int num_prms, double* prms,
  std::shared_ptr<CutoffFunction> cutfun)
{
  std::shared_ptr<TwoBodyDescriptor> descriptor;
  if (funtype == 0) descriptor = std::make_shared<BehlerG1>(num_prms, prms, cutfun);
  else if (funtype == 1) descriptor = std::make_shared<BehlerG2>(num_prms, prms, cutfun);
  else if (funtype == 2) descriptor = std::make_shared<BehlerG3>(num_prms, prms, cutfun);
// AUTOMATIC switch TwoBodyDescriptors start
  else if (funtype == 3) descriptor = std::make_shared<BehlerG1old>(num_prms, prms, cutfun);
  else if (funtype == 4) descriptor = std::make_shared<OneOverR6>(num_prms, prms, cutfun);
  else if (funtype == 5) descriptor = std::make_shared<OneOverR8>(num_prms, prms, cutfun);
  else if (funtype == 6) descriptor = std::make_shared<OneOverR10>(num_prms, prms, cutfun);
  else if (funtype == 7) descriptor = std::make_shared<radialTest>(num_prms, prms, cutfun);
// AUTOMATIC switch TwoBodyDescriptors end
  else printf("No function type %d\n", funtype);
  return descriptor;
}

std::shared_ptr<ThreeBodyDescriptor> switch_three_body_descriptors(
  int funtype, int num_prms, double* prms,
  std::shared_ptr<CutoffFunction> cutfun)
{
  std::shared_ptr<ThreeBodyDescriptor> descriptor;
  if (funtype == 0) descriptor = std::make_shared<BehlerG4>(num_prms, prms, cutfun);
  else if (funtype == 1) descriptor = std::make_shared<BehlerG5>(num_prms, prms, cutfun);
// AUTOMATIC switch ThreeBodyDescriptors start
  else if (funtype == 2) descriptor = std::make_shared<BehlerG4old>(num_prms, prms, cutfun);
  else if (funtype == 3) descriptor = std::make_shared<BehlerG5mod>(num_prms, prms, cutfun);
// AUTOMATIC switch ThreeBodyDescriptors end
  else printf("No function type %d\n", funtype);
  return descriptor;
}

int get_cutoff_function_by_name(const char* name)
{
  int id = -1;
  if (strcmp(name, "const") == 0) id = 0;
  else if (strcmp(name, "cos") == 0) id = 1;
  else if (strcmp(name, "tanh") == 0) id = 2;
  else if (strcmp(name, "polynomial") == 0) id = 3;
  else if (strcmp(name, "smooth") == 0) id = 4;
  else if (strcmp(name, "smooth2") == 0) id = 5;
  else if (strcmp(name, "shortRange") == 0) id = 6;
  else if (strcmp(name, "longRange") == 0) id = 7;
  return id;
}

int get_two_body_descriptor_by_name(const char* name)
{
  int id = -1;
  if (strcmp(name, "BehlerG1") == 0) id = 0;
  else if (strcmp(name, "BehlerG2") == 0) id = 1;
  else if (strcmp(name, "BehlerG3") == 0) id = 2;
// AUTOMATIC get_two_body_descriptor start
  else if (strcmp(name, "BehlerG1old") == 0) id = 3;
  else if (strcmp(name, "OneOverR6") == 0) id = 4;
  else if (strcmp(name, "OneOverR8") == 0) id = 5;
  else if (strcmp(name, "OneOverR10") == 0) id = 6;
  else if (strcmp(name, "radialTest") == 0) id = 7;
// AUTOMATIC get_two_body_descriptor end
  return id;
}

int get_three_body_descriptor_by_name(const char* name)
{
  int id = -1;
  if (strcmp(name, "BehlerG4") == 0) id = 0;
  else if (strcmp(name, "BehlerG5") == 0) id = 1;
// AUTOMATIC get_three_body_descriptor start
  else if (strcmp(name, "BehlerG4old") == 0) id = 2;
  else if (strcmp(name, "BehlerG5mod") == 0) id = 3;
// AUTOMATIC get_three_body_descriptor end
  return id;
}

void available_descriptors()
{
  printf("TwoBodyDescriptors: (key: name, # of parameters)\n");
  printf("0: BehlerG1, 0\n");
  printf("1: BehlerG2, 2\n");
  printf("2: BehlerG3, 1\n");
// AUTOMATIC available_two_body_descriptors start
  printf("3: BehlerG1old, 1\n");
  printf("4: OneOverR6, 0\n");
  printf("5: OneOverR8, 0\n");
  printf("6: OneOverR10, 0\n");
  printf("7: radialTest, 0\n");
// AUTOMATIC available_two_body_descriptors end
  printf("ThreeBodyDescriptors: (key: name, # of parameters)\n");
  printf("0: BehlerG4, 3\n");
  printf("1: BehlerG5, 3\n");
// AUTOMATIC available_three_body_descriptors start
  printf("2: BehlerG4old, 3\n");
  printf("3: BehlerG5mod, 5\n");
// AUTOMATIC available_three_body_descriptors end
}
