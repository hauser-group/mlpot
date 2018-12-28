#include <stdio.h>
#include "symmetryFunctionSet.h"

/* Compile using "g++ -g -std=c++11 test_for_leaks.cpp -I/path/to/NeuralNetworks/symmetryFunctions
-L/path/to/NeuralNetworks/symmetryFunctions -lSymFunSet -o test_for_leaks"

export LD_LIBRARY_PATH=$LD_LIBRARYPATH:/home/rmeyer/PythonModules/NeuralNetworks/NeuralNetworks/descriptors*/


int main()
{
  printf("Programm started\n");
  SymmetryFunctionSet* sfs = new SymmetryFunctionSet(2);
  double* prms = new double[2] {0.0, 1.0};
  double* prms3 = new double[3] {1.0, 1.0, 1.0};
  sfs->add_TwoBodySymmetryFunction(0, 0, 0, 2, prms, 0, 7.0);
  sfs->add_TwoBodySymmetryFunction(0, 1, 0, 2, prms, 0, 7.0);
  sfs->add_TwoBodySymmetryFunction(1, 0, 0, 2, prms, 0, 7.0);
  sfs->add_TwoBodySymmetryFunction(1, 1, 0, 2, prms, 0, 7.0);
  sfs->add_ThreeBodySymmetryFunction(0, 0, 0, 0, 3, prms3, 0, 7.0);
  sfs->add_ThreeBodySymmetryFunction(0, 0, 1, 0, 3, prms3, 0, 7.0);
  sfs->add_ThreeBodySymmetryFunction(0, 1, 1, 0, 3, prms3, 0, 7.0);
  sfs->add_ThreeBodySymmetryFunction(1, 0, 0, 0, 3, prms3, 0, 7.0);
  sfs->add_ThreeBodySymmetryFunction(1, 0, 1, 0, 3, prms3, 0, 7.0);
  sfs->add_ThreeBodySymmetryFunction(1, 1, 1, 0, 3, prms3, 0, 7.0);
  printf("SymFunSet created\n");

  int num_atoms = 20;
  int* types = new int[num_atoms]();
  types[2] = 1;
  types[3] = 1;

  int G_size = sfs->get_G_vector_size(num_atoms, types);
  double* G_vector = new double[G_size]();
  double* dG_tensor = new double[G_size*3*num_atoms]();
  double* xyzs = new double[3*num_atoms]();

  sfs->eval(num_atoms, types, xyzs, G_vector);
  sfs->eval_derivatives(num_atoms, types, xyzs, dG_tensor);

  sfs->eval_with_derivatives(num_atoms, types, xyzs, G_vector, dG_tensor);

  sfs->eval_atomwise(num_atoms, types, xyzs, G_vector);
  sfs->eval_derivatives_atomwise(num_atoms, types, xyzs, dG_tensor);

  sfs->eval_with_derivatives_atomwise(num_atoms, types, xyzs, G_vector, dG_tensor);

  delete[] prms;
  delete[] prms3;
  delete[] types;
  delete[] G_vector;
  delete[] dG_tensor;
  delete[] xyzs;
  delete sfs;
  return 0;
}
