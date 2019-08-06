#include "symmetryFunctions.h"
#include <vector>
#include <memory>

class DescriptorSet
{
  public:
    DescriptorSet(int num_atomtypes_i);
    ~DescriptorSet();
    void add_two_body_descriptor(
      int type1, int type2, int funtype, int num_prms, double* prms,
      int cutoff_type, double cutoff);
    void add_three_body_descriptor(
      int type1, int type2, int type3, int funtype, int num_prms,
      double* prms, int cutoff_type, double cutoff);

    int get_G_vector_size(int num_atoms, int* types);

    void eval(int num_atoms, int* types, double* xyzs, double* G_vector);
    void eval_atomwise(
      int num_atoms, int* types, double* xyzs, double* G_vector);

    void eval_derivatives(
      int num_atoms, int* types, double* xyzs, double* dG_tensor);
    void eval_derivatives_atomwise(
      int num_atoms, int* types, double* xyzs, double* dG_tensor);

    void eval_with_derivatives(int num_atoms, int* types, double* xyzs,
      double* G_vector, double* dG_tensor);
    void eval_with_derivatives_atomwise(int num_atoms, int* types, double* xyzs,
      double* G_vector, double* dG_tensor);
    void print_descriptors() const;

  private:
    int num_atomtypes, num_atomtypes_sq;
    std::vector<int> num_descriptors;
    std::vector <std::vector<std::shared_ptr<TwoBodySymmetryFunction> > >
      two_body_descriptors;
    std::vector<int> pos_two_body;
    std::vector <std::vector<std::shared_ptr<ThreeBodySymmetryFunction> > >
      three_body_descriptors;
    std::vector<int> pos_three_body;
    std::vector<double> max_cutoff;
    double global_max_cutoff;
};

// Wrap the C++ classes for C usage in python ctypes:
extern "C" {
  DescriptorSet* create_descriptor_set(int num_atomtypes)
  {
    return new DescriptorSet(num_atomtypes);
  }
  void destroy_descriptor_set(DescriptorSet* ds)
  {
    delete ds;
  }
  void descriptor_set_add_two_body_descriptor(
    DescriptorSet* ds,  int type1, int type2, int funtype,
    int num_prms, double* prms, int cutoff_type, double cutoff)
  {
    ds->add_two_body_descriptor(type1, type2, funtype,
      num_prms, prms, cutoff_type, cutoff);
  }
  void descriptor_set_add_three_body_descriptor(
    DescriptorSet* ds,  int type1, int type2, int type3,
    int funtype, int num_prms, double* prms, int cutoff_type, double cutoff)
  {
    ds->add_three_body_descriptor(type1, type2, type3, funtype,
      num_prms, prms, cutoff_type, cutoff);
  }
  void descriptor_set_print_descriptors(DescriptorSet* ds){
    ds->print_descriptors();
  }
  void descriptor_set_available_descriptors(){
    available_symFuns();
  }
  int descriptor_set_get_cutoff_function_by_name(const char* name)
  {
    return descriptor_set_get_cutoff_function_by_name(name);
  }
  int descriptor_set_get_two_body_descriptor_by_name(const char* name)
  {
    return descriptor_set_get_two_body_descriptor_by_name(name);
  }
  int descriptor_set_get_three_body_descriptor_by_name(const char* name)
  {
    return descriptor_set_get_three_body_descriptor_by_name(name);
  }
  int descriptor_set_get_G_vector_size(DescriptorSet* ds,
    int num_atoms, int* types)
  {
    return ds->get_G_vector_size(num_atoms, types);
  }
  void descriptor_set_eval(DescriptorSet* ds, int num_atoms,
    int* types, double* xyzs, double* G_vector)
  {
    ds->eval(num_atoms, types, xyzs, G_vector);
  }
  void descriptor_set_eval_derivatives(DescriptorSet* ds,
    int num_atoms, int* types, double* xyzs, double* dG_tensor)
  {
    ds->eval_derivatives(num_atoms, types, xyzs, dG_tensor);
  }
  void descriptor_set_eval_with_derivatives(
    DescriptorSet* ds, int num_atoms, int* types, double* xyzs,
    double* G_vector, double* dG_tensor)
  {
    ds->eval_with_derivatives(
      num_atoms, types, xyzs, G_vector, dG_tensor);
  }
  void descriptor_set_eval_atomwise(DescriptorSet* ds,
    int num_atoms, int* types, double* xyzs, double* G_vector)
  {
    ds->eval_atomwise(num_atoms, types, xyzs, G_vector);
  }
  void descriptor_set_eval_derivatives_atomwise(DescriptorSet* ds,
    int num_atoms, int* types, double* xyzs, double* dG_tensor)
  {
    ds->eval_derivatives_atomwise(num_atoms, types, xyzs, dG_tensor);
  }
  void descriptor_set_eval_with_derivatives_atomwise(DescriptorSet* ds,
    int num_atoms, int* types, double* xyzs, double* G_vector, double* dG_tensor)
  {
    ds->eval_with_derivatives_atomwise(num_atoms, types, xyzs, G_vector, dG_tensor);
  }
};
