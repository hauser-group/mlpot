/*

CAUTION: Part of this file is written by the python script generateSymFuns.py.
These parts are marked by comment lines starting with AUTOMATIC. Do not alter
anything between these tags.
*/

#include "symmetryFunctions.h"
#include <stdio.h>
#include <math.h>
#include <limits>
#include <string.h>

SymmetryFunction::SymmetryFunction(int num_prms_i, double* prms_i,
  std::shared_ptr<CutoffFunction> cutfun_i):cutfun(cutfun_i)
{
  num_prms = num_prms_i;
  prms = new double[num_prms];
  for (int i = 0; i < num_prms; i++)
  {
    prms[i] = prms_i[i];
  }
};

SymmetryFunction::~SymmetryFunction()
{
  delete[] prms;
};

SymmetryFunction::SymmetryFunction(const SymmetryFunction& other) //Copy constructor
{
  num_prms = other.num_prms;
  prms = new double[other.num_prms];
  for (int i = 0; i < num_prms; i++)
  {
    prms[i] = other.prms[i];
  }
};

SymmetryFunction& SymmetryFunction::operator=(const SymmetryFunction& other) //Copy assignment
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

// AUTOMATIC Start of custom TwoBodySymFuns

double BehlerG0::eval(double rij)
{
  return cutfun->eval(rij);
};

double BehlerG0::drij(double rij)
{
  return cutfun->derivative(rij);
};

void BehlerG0::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  G = cutfun->eval(rij);
  dGdrij = cutfun->derivative(rij);
};

double BehlerG1::eval(double rij)
{
  return cutfun->eval(rij)*exp(-prms[0]*pow(rij, 2));
};

double BehlerG1::drij(double rij)
{
  return (-2*prms[0]*rij*cutfun->eval(rij) + cutfun->derivative(rij))*exp(-prms[0]*pow(rij, 2));
};

void BehlerG1::eval_with_derivatives(double rij, double &G, double &dGdrij)
{
  auto x0 = cutfun->eval(rij);
  auto x1 = exp(-prms[0]*pow(rij, 2));
  G = x0*x1;
  dGdrij = x1*(-2*prms[0]*rij*x0 + cutfun->derivative(rij));
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
// AUTOMATIC End of custom TwoBodySymFuns

// AUTOMATIC Start of custom ThreeBodySymFuns

double BehlerG3::eval(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*exp2(-prms[1] + 1)*cutfun->eval(rij)*cutfun->eval(rik)*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*exp(-prms[2]*(-2*costheta*rij*rik + 2*pow(rij, 2) + 2*pow(rik, 2)));
};

double BehlerG3::drij(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*((-costheta*rik + rij)*cutfun->derivative(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*cutfun->eval(rij) + (2*prms[2]*(costheta*rik - 2*rij)*cutfun->eval(rij) + cutfun->derivative(rij))*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))))*exp2(-prms[1] + 1)*cutfun->eval(rik)*exp(-2*prms[2]*(-costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))/sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2));
};

double BehlerG3::drik(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*((-costheta*rij + rik)*cutfun->derivative(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))*cutfun->eval(rik) + (2*prms[2]*(costheta*rij - 2*rik)*cutfun->eval(rik) + cutfun->derivative(rik))*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))))*exp2(-prms[1] + 1)*cutfun->eval(rij)*exp(-2*prms[2]*(-costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))/sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2));
};

double BehlerG3::dcostheta(double rij, double rik, double costheta)
{
  return (prms[0]*prms[1]*pow(costheta*prms[0] + 1, prms[1])*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))) + 2*prms[2]*rij*rik*pow(costheta*prms[0] + 1, prms[1] + 1)*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))*cutfun->eval(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))) - rij*rik*pow(costheta*prms[0] + 1, prms[1] + 1)*cutfun->derivative(sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2))))*exp2(-prms[1] + 1)*cutfun->eval(rij)*cutfun->eval(rik)*exp(-2*prms[2]*(-costheta*rij*rik + pow(rij, 2) + pow(rik, 2)))/((costheta*prms[0] + 1)*sqrt(-2*costheta*rij*rik + pow(rij, 2) + pow(rik, 2)));
};

void BehlerG3::eval_with_derivatives(double rij, double rik, double costheta,
  double &G, double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = cutfun->eval(rik);
  auto x1 = 2*rij;
  auto x2 = costheta*rik;
  auto x3 = pow(rij, 2) + pow(rik, 2);
  auto x4 = sqrt(-x1*x2 + x3);
  auto x5 = cutfun->eval(x4);
  auto x6 = cutfun->eval(rij);
  auto x7 = exp2(-prms[1] + 1);
  auto x8 = costheta*prms[0] + 1;
  auto x9 = pow(x8, prms[1]);
  auto x10 = 2*prms[2];
  auto x11 = costheta*rij;
  auto x12 = exp(-x10*(-rik*x11 + x3));
  auto x13 = x12*x6*x7*x9;
  auto x14 = 1.0/x4;
  auto x15 = x0*x12*x14*x7;
  auto x16 = cutfun->derivative(x4);
  auto x17 = x4*x5;
  auto x18 = pow(x8, prms[1] + 1);
  G = x0*x13*x5;
  dGdrij = x15*x9*(x16*x6*(rij - x2) + x17*(x10*x6*(-x1 + x2) + cutfun->derivative(rij)));
  dGdrik = x13*x14*(x0*x16*(rik - x11) + x17*(x0*x10*(-2*rik + x11) + cutfun->derivative(rik)));
  dGdcostheta = x15*x6*(prms[0]*prms[1]*x17*x9 + 2*prms[2]*rij*rik*x17*x18 - rij*rik*x16*x18)/x8;
};

void BehlerG3::derivatives(double rij, double rik, double costheta,
  double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = exp2(-prms[1] + 1);
  auto x1 = costheta*prms[0] + 1;
  auto x2 = pow(x1, prms[1]);
  auto x3 = 2*rij;
  auto x4 = costheta*rik;
  auto x5 = pow(rij, 2) + pow(rik, 2);
  auto x6 = sqrt(-x3*x4 + x5);
  auto x7 = 1.0/x6;
  auto x8 = 2*prms[2];
  auto x9 = costheta*rij;
  auto x10 = exp(-x8*(-rik*x9 + x5));
  auto x11 = x0*x10*x2*x7;
  auto x12 = cutfun->eval(rik);
  auto x13 = cutfun->eval(rij);
  auto x14 = cutfun->derivative(x6);
  auto x15 = x6*cutfun->eval(x6);
  auto x16 = pow(x1, prms[1] + 1);
  dGdrij = x11*x12*(x13*x14*(rij - x4) + x15*(x13*x8*(-x3 + x4) + cutfun->derivative(rij)));
  dGdrik = x11*x13*(x12*x14*(rik - x9) + x15*(x12*x8*(-2*rik + x9) + cutfun->derivative(rik)));
  dGdcostheta = x0*x10*x12*x13*x7*(prms[0]*prms[1]*x15*x2 + 2*prms[2]*rij*rik*x15*x16 - rij*rik*x14*x16)/x1;
};

double BehlerG4::eval(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*exp2(-prms[1] + 1)*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG4::drij(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-2*prms[2]*rij*cutfun->eval(rij) + cutfun->derivative(rij))*exp2(-prms[1] + 1)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG4::drik(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-2*prms[2]*rik*cutfun->eval(rik) + cutfun->derivative(rik))*exp2(-prms[1] + 1)*cutfun->eval(rij)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

double BehlerG4::dcostheta(double rij, double rik, double costheta)
{
  return prms[0]*prms[1]*pow(costheta*prms[0] + 1, prms[1] - 1)*exp2(-prms[1] + 1)*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)));
};

void BehlerG4::eval_with_derivatives(double rij, double rik, double costheta,
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

void BehlerG4::derivatives(double rij, double rik, double costheta,
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

double MeyerG1::eval(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*exp2(-prms[1] + 1)*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2)))*exp(-prms[2]*(pow(-prms[3] + rij, 2) + pow(-prms[3] + rik, 2)));
};

double MeyerG1::drij(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-2*prms[2]*rij*cutfun->eval(rij) + 2*prms[2]*(prms[3] - rij)*cutfun->eval(rij) + cutfun->derivative(rij))*exp2(-prms[1] + 1)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2) + pow(prms[3] - rij, 2) + pow(prms[3] - rik, 2)));
};

double MeyerG1::drik(double rij, double rik, double costheta)
{
  return pow(costheta*prms[0] + 1, prms[1])*(-2*prms[2]*rik*cutfun->eval(rik) + 2*prms[2]*(prms[3] - rik)*cutfun->eval(rik) + cutfun->derivative(rik))*exp2(-prms[1] + 1)*cutfun->eval(rij)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2) + pow(prms[3] - rij, 2) + pow(prms[3] - rik, 2)));
};

double MeyerG1::dcostheta(double rij, double rik, double costheta)
{
  return prms[0]*prms[1]*pow(costheta*prms[0] + 1, prms[1] - 1)*exp2(-prms[1] + 1)*cutfun->eval(rij)*cutfun->eval(rik)*exp(-prms[2]*(pow(rij, 2) + pow(rik, 2) + pow(prms[3] - rij, 2) + pow(prms[3] - rik, 2)));
};

void MeyerG1::eval_with_derivatives(double rij, double rik, double costheta,
  double &G, double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = costheta*prms[0] + 1;
  auto x1 = pow(x0, prms[1]);
  auto x2 = cutfun->eval(rij);
  auto x3 = cutfun->eval(rik);
  auto x4 = exp2(-prms[1] + 1);
  auto x5 = prms[3] - rij;
  auto x6 = prms[3] - rik;
  auto x7 = exp(-prms[2]*(pow(rij, 2) + pow(rik, 2) + pow(x5, 2) + pow(x6, 2)));
  auto x8 = x2*x3*x4*x7;
  auto x9 = 2*prms[2]*x2;
  auto x10 = x1*x4*x7;
  auto x11 = 2*prms[2]*x3;
  G = x1*x8;
  dGdrij = x10*x3*(-rij*x9 + x5*x9 + cutfun->derivative(rij));
  dGdrik = x10*x2*(-rik*x11 + x11*x6 + cutfun->derivative(rik));
  dGdcostheta = prms[0]*prms[1]*pow(x0, prms[1] - 1)*x8;
};

void MeyerG1::derivatives(double rij, double rik, double costheta,
  double &dGdrij, double &dGdrik, double &dGdcostheta)
{
  auto x0 = costheta*prms[0] + 1;
  auto x1 = pow(x0, prms[1]);
  auto x2 = cutfun->eval(rij);
  auto x3 = 2*prms[2]*x2;
  auto x4 = prms[3] - rij;
  auto x5 = cutfun->eval(rik);
  auto x6 = exp2(-prms[1] + 1);
  auto x7 = prms[3] - rik;
  auto x8 = exp(-prms[2]*(pow(rij, 2) + pow(rik, 2) + pow(x4, 2) + pow(x7, 2)));
  auto x9 = x5*x6*x8;
  auto x10 = 2*prms[2]*x5;
  dGdrij = x1*x9*(-rij*x3 + x3*x4 + cutfun->derivative(rij));
  dGdrik = x1*x2*x6*x8*(-rik*x10 + x10*x7 + cutfun->derivative(rik));
  dGdcostheta = prms[0]*prms[1]*pow(x0, prms[1] - 1)*x2*x9;
};
// AUTOMATIC End of custom ThreeBodySymFuns

std::shared_ptr<CutoffFunction> switch_CutFun(
  int cutoff_type, double cutoff)
{
  std::shared_ptr<CutoffFunction> cutfun;
  switch (cutoff_type) {
    case 0:
      cutfun = std::make_shared<ConstCutoffFunction>(cutoff);
      break;
    case 1:
      cutfun = std::make_shared<CosCutoffFunction>(cutoff);
      break;
    case 2:
      cutfun = std::make_shared<TanhCutoffFunction>(cutoff);
      break;
    case 3:
      cutfun = std::make_shared<PolynomialCutoffFunction>(cutoff);
      break;
    case 4:
      cutfun = std::make_shared<SmoothCutoffFunction>(cutoff);
      break;
    case 5:
      cutfun = std::make_shared<ShortRangeCutoffFunction>(cutoff);
      break;
    case 6:
      cutfun = std::make_shared<LongRangeCutoffFunction>(cutoff);
      break;
  }
  return cutfun;
}

std::shared_ptr<TwoBodySymmetryFunction> switch_TwoBodySymFun(int funtype,
  int num_prms, double* prms, std::shared_ptr<CutoffFunction> cutfun)
{
  std::shared_ptr<TwoBodySymmetryFunction> symFun;
  switch (funtype){
// AUTOMATIC TwoBodySymmetryFunction switch start
    case 0:
      symFun = std::make_shared<BehlerG0>(num_prms, prms, cutfun);
      break;
    case 1:
      symFun = std::make_shared<BehlerG1>(num_prms, prms, cutfun);
      break;
    case 2:
      symFun = std::make_shared<BehlerG2>(num_prms, prms, cutfun);
      break;
    case 3:
      symFun = std::make_shared<OneOverR6>(num_prms, prms, cutfun);
      break;
    case 4:
      symFun = std::make_shared<OneOverR8>(num_prms, prms, cutfun);
      break;
    case 5:
      symFun = std::make_shared<OneOverR10>(num_prms, prms, cutfun);
      break;
    case 6:
      symFun = std::make_shared<radialTest>(num_prms, prms, cutfun);
      break;
// AUTOMATIC TwoBodySymmetryFunction switch end
    default:
      printf("No function type %d\n", funtype);
  }
  return symFun;
}

std::shared_ptr<ThreeBodySymmetryFunction> switch_ThreeBodySymFun(int funtype,
  int num_prms, double* prms, std::shared_ptr<CutoffFunction> cutfun)
{
  std::shared_ptr<ThreeBodySymmetryFunction> symFun;
  switch (funtype){
// AUTOMATIC ThreeBodySymmetryFunction switch start
    case 0:
      symFun = std::make_shared<BehlerG3>(num_prms, prms, cutfun);
      break;
    case 1:
      symFun = std::make_shared<BehlerG4>(num_prms, prms, cutfun);
      break;
    case 2:
      symFun = std::make_shared<MeyerG1>(num_prms, prms, cutfun);
      break;
// AUTOMATIC ThreeBodySymmetryFunction switch end
    default:
      printf("No function type %d\n", funtype);
  }
  return symFun;
}

int get_CutFun_by_name(const char* name)
{
  int id = -1;
  if (strcmp(name, "const") == 0)
  {
    id = 0;
  } else if (strcmp(name, "cos") == 0)
  {
    id = 1;
  } else if (strcmp(name, "tanh") == 0)
  {
    id = 2;
  } else if (strcmp(name, "polynomial") == 0)
  {
    id = 3;
  } else if (strcmp(name, "smooth") == 0)
  {
    id = 4;
  } else if (strcmp(name, "shortRange") == 0)
  {
    id = 5;
  } else if (strcmp(name, "longRange") == 0)
  {
    id = 6;
  }
  return id;
}

int get_TwoBodySymFun_by_name(const char* name)
{
  int id = -1;
// AUTOMATIC get_TwoBodySymFuns start
  if (strcmp(name, "BehlerG0") == 0)
  {
    id = 0;
  }
  if (strcmp(name, "BehlerG1") == 0)
  {
    id = 1;
  }
  if (strcmp(name, "BehlerG2") == 0)
  {
    id = 2;
  }
  if (strcmp(name, "OneOverR6") == 0)
  {
    id = 3;
  }
  if (strcmp(name, "OneOverR8") == 0)
  {
    id = 4;
  }
  if (strcmp(name, "OneOverR10") == 0)
  {
    id = 5;
  }
  if (strcmp(name, "radialTest") == 0)
  {
    id = 6;
  }
// AUTOMATIC get_TwoBodySymFuns end
  return id;
}

int get_ThreeBodySymFun_by_name(const char* name)
{
  int id = -1;
// AUTOMATIC get_ThreeBodySymFuns start
  if (strcmp(name, "BehlerG3") == 0)
  {
    id = 0;
  }
  if (strcmp(name, "BehlerG4") == 0)
  {
    id = 1;
  }
  if (strcmp(name, "MeyerG1") == 0)
  {
    id = 2;
  }
// AUTOMATIC get_ThreeBodySymFuns end
  return id;
}

void available_symFuns()
{
// AUTOMATIC available_symFuns start
  printf("TwoBodySymmetryFunctions: (key: name, # of parameters)\n");
  printf("0: BehlerG0, 0\n");
  printf("1: BehlerG1, 1\n");
  printf("2: BehlerG2, 2\n");
  printf("3: OneOverR6, 0\n");
  printf("4: OneOverR8, 0\n");
  printf("5: OneOverR10, 0\n");
  printf("6: radialTest, 0\n");
  printf("ThreeBodySymmetryFunctions: (key: name, # of parameters)\n");
  printf("0: BehlerG3, 3\n");
  printf("1: BehlerG4, 3\n");
  printf("2: MeyerG1, 4\n");
// AUTOMATIC available_symFuns end
}
