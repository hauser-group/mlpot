#include "symmetryFunctions.h"
#include <vector>
#include <memory>

class SymmetryFunctionSet
{
  public:
    SymmetryFunctionSet(int num_atomtypes_i);
    ~SymmetryFunctionSet();
    void add_TwoBodySymmetryFunction(
      int type1, int type2, int funtype, int num_prms, double* prms,
      int cutoff_type, double cutoff);
    void add_ThreeBodySymmetryFunction(
      int type1, int type2, int type3, int funtype, int num_prms,
      double* prms, int cutoff_type, double cutoff);

    int get_G_vector_size(int num_atoms, int* types);

    void eval(int num_atoms, int* types, double* xyzs, double* G_vector);
    void eval_old(int num_atoms, int* types, double* xyzs, double* G_vector);

    void eval_derivatives(
      int num_atoms, int* types, double* xyzs, double* dG_tensor);
    void eval_derivatives_old(
      int num_atoms, int* types, double* xyzs, double* dG_tensor);
    void eval_with_derivatives(int num_atoms, int* types, double* xyzs,
      double* G_vector, double* dG_tensor);
    void print_symFuns() const;

  private:
    int num_atomtypes, num_atomtypes_sq;
    int* num_symFuns;
    std::vector <std::vector<std::shared_ptr<TwoBodySymmetryFunction> > >
      twoBodySymFuns;
    int* pos_twoBody;
    std::vector <std::vector<std::shared_ptr<ThreeBodySymmetryFunction> > >
      threeBodySymFuns;
    int* pos_threeBody;
    double* max_cutoff;
};

// Wrap the C++ classes for C usage in python ctypes:
extern "C" {
  SymmetryFunctionSet* create_SymmetryFunctionSet(int num_atomtypes)
  {
    return new SymmetryFunctionSet(num_atomtypes);
  }
  void destroy_SymmetryFunctionSet(SymmetryFunctionSet* symFunSet)
  {
    delete symFunSet;
  }
  void SymmetryFunctionSet_add_TwoBodySymmetryFunction(
    SymmetryFunctionSet* symFunSet,  int type1, int type2, int funtype,
    int num_prms, double* prms, int cutoff_type, double cutoff)
  {
    symFunSet->add_TwoBodySymmetryFunction(type1, type2, funtype,
      num_prms, prms, cutoff_type, cutoff);
  }
  void SymmetryFunctionSet_add_ThreeBodySymmetryFunction(
    SymmetryFunctionSet* symFunSet,  int type1, int type2, int type3,
    int funtype, int num_prms, double* prms, int cutoff_type, double cutoff)
  {
    symFunSet->add_ThreeBodySymmetryFunction(type1, type2, type3, funtype,
      num_prms, prms, cutoff_type, cutoff);
  }
  void SymmetryFunctionSet_print_symFuns(SymmetryFunctionSet* symFunSet){
    symFunSet->print_symFuns();
  }
  void SymmetryFunctionSet_available_symFuns(){
    available_symFuns();
  }
  int SymmetryFunctionSet_get_CutFun_by_name(const char* name)
  {
    return get_CutFun_by_name(name);
  }
  int SymmetryFunctionSet_get_TwoBodySymFun_by_name(const char* name)
  {
    return get_TwoBodySymFun_by_name(name);
  }
  int SymmetryFunctionSet_get_ThreeBodySymFun_by_name(const char* name)
  {
    return get_ThreeBodySymFun_by_name(name);
  }
  int SymmetryFunctionSet_get_G_vector_size(SymmetryFunctionSet* symFunSet,
    int num_atoms, int* types)
  {
    return symFunSet->get_G_vector_size(num_atoms, types);
  }
  void SymmetryFunctionSet_eval(SymmetryFunctionSet* symFunSet, int num_atoms,
    int* types, double* xyzs, double* G_vector)
  {
    symFunSet->eval(num_atoms, types, xyzs, G_vector);
  }
  void SymmetryFunctionSet_eval_derivatives(SymmetryFunctionSet* symFunSet,
    int num_atoms, int* types, double* xyzs, double* dG_tensor)
  {
    symFunSet->eval_derivatives(num_atoms, types, xyzs, dG_tensor);
  }
  void SymmetryFunctionSet_eval_with_derivatives(
    SymmetryFunctionSet* symFunSet, int num_atoms, int* types, double* xyzs,
    double* G_vector, double* dG_tensor)
  {
    symFunSet->eval_with_derivatives(
      num_atoms, types, xyzs, G_vector, dG_tensor);
  }
  void SymmetryFunctionSet_eval_old(SymmetryFunctionSet* symFunSet,
    int num_atoms, int* types, double* xyzs, double* G_vector)
  {
    symFunSet->eval_old(num_atoms, types, xyzs, G_vector);
  }
  void SymmetryFunctionSet_eval_derivatives_old(SymmetryFunctionSet* symFunSet,
    int num_atoms, int* types, double* xyzs, double* dG_tensor)
  {
    symFunSet->eval_derivatives_old(num_atoms, types, xyzs, dG_tensor);
  }
};
