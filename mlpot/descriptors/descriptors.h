/*

CAUTION: Part of this file is written by the python script
generate_custom_descriptors.py.
These parts are marked by comment lines starting with AUTOMATIC. Do not alter
anything between these tags.
*/

#include "cutoff_functions.h"
#include <memory>

class Descriptor
{
    public:
        Descriptor(int num_prms, double* pmrs,
          std::shared_ptr<CutoffFunction> cutfun);
        ~Descriptor();
        Descriptor(const Descriptor& other);
        Descriptor& operator=(const Descriptor& other);
    protected:
        int num_prms;
        double* prms;
        std::shared_ptr<CutoffFunction> cutfun;
};

class TwoBodyDescriptor: public Descriptor
{
    public:
        TwoBodyDescriptor(int num_prms, double* prms,
          std::shared_ptr<CutoffFunction> cutfun):
          Descriptor(num_prms, prms, cutfun){};
        virtual double eval(double rij) = 0;
        virtual double drij(double rij) = 0;
        virtual void eval_with_derivatives(
          double rij, double &G, double &dGdrij) = 0;
};

class BehlerG1: public TwoBodyDescriptor
{
    public:
        BehlerG1(int num_prms, double* prms,
          std::shared_ptr<CutoffFunction> cutfun):
          TwoBodyDescriptor(num_prms, prms, cutfun){};
        double eval(double rij);
        double drij(double rij);
        void eval_with_derivatives(double rij, double &G, double &dGdrij);
};

class BehlerG2: public TwoBodyDescriptor
{
    public:
        BehlerG2(int num_prms, double* prms,
          std::shared_ptr<CutoffFunction> cutfun):
          TwoBodyDescriptor(num_prms, prms, cutfun){};
        double eval(double rij);
        double drij(double rij);
        void eval_with_derivatives(double rij, double &G, double &dGdrij);
};

class BehlerG3: public TwoBodyDescriptor
{
    public:
        BehlerG3(int num_prms, double* prms,
          std::shared_ptr<CutoffFunction> cutfun):
          TwoBodyDescriptor(num_prms, prms, cutfun){};
        double eval(double rij);
        double drij(double rij);
        void eval_with_derivatives(double rij, double &G, double &dGdrij);
};

// AUTOMATIC custom TwoBodyDescriptors start

class BehlerG1old: public TwoBodyDescriptor
{
    public:
        BehlerG1old(int num_prms, double* prms,
          std::shared_ptr<CutoffFunction> cutfun):
          TwoBodyDescriptor(num_prms, prms, cutfun){};
        double eval(double rij);
        double drij(double rij);
        void eval_with_derivatives(double rij, double &G, double &dGdrij);
};

class OneOverR6: public TwoBodyDescriptor
{
    public:
        OneOverR6(int num_prms, double* prms,
          std::shared_ptr<CutoffFunction> cutfun):
          TwoBodyDescriptor(num_prms, prms, cutfun){};
        double eval(double rij);
        double drij(double rij);
        void eval_with_derivatives(double rij, double &G, double &dGdrij);
};

class OneOverR8: public TwoBodyDescriptor
{
    public:
        OneOverR8(int num_prms, double* prms,
          std::shared_ptr<CutoffFunction> cutfun):
          TwoBodyDescriptor(num_prms, prms, cutfun){};
        double eval(double rij);
        double drij(double rij);
        void eval_with_derivatives(double rij, double &G, double &dGdrij);
};

class OneOverR10: public TwoBodyDescriptor
{
    public:
        OneOverR10(int num_prms, double* prms,
          std::shared_ptr<CutoffFunction> cutfun):
          TwoBodyDescriptor(num_prms, prms, cutfun){};
        double eval(double rij);
        double drij(double rij);
        void eval_with_derivatives(double rij, double &G, double &dGdrij);
};

class radialTest: public TwoBodyDescriptor
{
    public:
        radialTest(int num_prms, double* prms,
          std::shared_ptr<CutoffFunction> cutfun):
          TwoBodyDescriptor(num_prms, prms, cutfun){};
        double eval(double rij);
        double drij(double rij);
        void eval_with_derivatives(double rij, double &G, double &dGdrij);
};
// AUTOMATIC custom TwoBodyDescriptors end

class ThreeBodyDescriptor: public Descriptor
{
    public:
      ThreeBodyDescriptor(int num_prms, double* prms,
        std::shared_ptr<CutoffFunction> cutfun):
        Descriptor(num_prms, prms, cutfun){};
      virtual double eval(double rij, double rik, double costheta) = 0;
      virtual double drij(double rij, double rik, double costheta) = 0;
      virtual double drik(double rij, double rik, double costheta) = 0;
      virtual double dcostheta(double rij, double rik, double costheta) = 0;
      virtual void derivatives(double rij, double rik, double costheta,
        double &dGdrij, double &dGdrik, double &dGdcostheta) = 0;
      virtual void eval_with_derivatives(
        double rij, double rik, double costheta,
        double &G, double &dGdrij, double &dGdrik, double &dGdcostheta) = 0;
};

class BehlerG4: public ThreeBodyDescriptor
{
  public:
    BehlerG4(int num_prms, double* prms,
      std::shared_ptr<CutoffFunction> cutfun):
      ThreeBodyDescriptor(num_prms, prms, cutfun){};
    double eval(double rij, double rik, double costheta);
    double drij(double rij, double rik, double costheta);
    double drik(double rij, double rik, double costheta);
    double dcostheta(double rij, double rik, double costheta);
    void derivatives(double rij, double rik, double costheta,
      double &dGdrij, double &dGdrik, double &dGdcostheta);
    void eval_with_derivatives(double rij, double rik, double costheta,
      double &G, double &dGdrij, double &dGdrik, double &dGdcostheta);
};

class BehlerG5: public ThreeBodyDescriptor
{
  public:
    BehlerG5(int num_prms, double* prms,
      std::shared_ptr<CutoffFunction> cutfun):
      ThreeBodyDescriptor(num_prms, prms, cutfun){};
    double eval(double rij, double rik, double costheta);
    double drij(double rij, double rik, double costheta);
    double drik(double rij, double rik, double costheta);
    double dcostheta(double rij, double rik, double costheta);
    void derivatives(double rij, double rik, double costheta,
      double &dGdrij, double &dGdrik, double &dGdcostheta);
    void eval_with_derivatives(double rij, double rik, double costheta,
      double &G, double &dGdrij, double &dGdrik, double &dGdcostheta);
};

// AUTOMATIC custom ThreeBodyDescriptors start

class BehlerG4auto: public ThreeBodyDescriptor
{
  public:
    BehlerG4auto(int num_prms, double* prms,
      std::shared_ptr<CutoffFunction> cutfun):
      ThreeBodyDescriptor(num_prms, prms, cutfun){};
    double eval(double rij, double rik, double costheta);
    double drij(double rij, double rik, double costheta);
    double drik(double rij, double rik, double costheta);
    double dcostheta(double rij, double rik, double costheta);
    void derivatives(double rij, double rik, double costheta,
      double &dGdrij, double &dGdrik, double &dGdcostheta);
    void eval_with_derivatives(double rij, double rik, double costheta,
      double &G, double &dGdrij, double &dGdrik, double &dGdcostheta);
};

class BehlerG5mod: public ThreeBodyDescriptor
{
  public:
    BehlerG5mod(int num_prms, double* prms,
      std::shared_ptr<CutoffFunction> cutfun):
      ThreeBodyDescriptor(num_prms, prms, cutfun){};
    double eval(double rij, double rik, double costheta);
    double drij(double rij, double rik, double costheta);
    double drik(double rij, double rik, double costheta);
    double dcostheta(double rij, double rik, double costheta);
    void derivatives(double rij, double rik, double costheta,
      double &dGdrij, double &dGdrik, double &dGdcostheta);
    void eval_with_derivatives(double rij, double rik, double costheta,
      double &G, double &dGdrij, double &dGdrik, double &dGdcostheta);
};
// AUTOMATIC custom ThreeBodyDescriptors end

std::shared_ptr<CutoffFunction> switch_cutoff_functions(
  int cutoff_type, double cutoff);
std::shared_ptr<TwoBodyDescriptor> switch_two_body_descriptors(
  int funtype, int num_prms, double* prms,
  std::shared_ptr<CutoffFunction> cutfun);
std::shared_ptr<ThreeBodyDescriptor> switch_three_body_descriptors(
  int funtype, int num_prms, double* prms,
  std::shared_ptr<CutoffFunction> cutfun);
int get_cutoff_function_by_name(const char* name);
int get_two_body_descriptor_by_name(const char* name);
int get_three_body_descriptor_by_name(const char* name);
void available_descriptors();
