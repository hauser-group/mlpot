#include <Python.h>
#include "symmetryFunctionSet.h"

/* Follows the example given at:
   http://www.speedupcode.com/c-class-in-python3/
*/

PyObject* construct(PyObject* self, PyObject* args)
{
  int num_atomtypes;

  PyArg_ParseTuple(args, "i", &num_atomtypes); // "i" for integer

  symmetryFunctionSet* ds = new symmetryFunctionSet(num_atomtypes);

  PyObject descriptorsetCapsule = PyCapsule_New((void *)ds, "DescribSetPtr",
                                                 NULL);
  PyCapsule_SetPointer(descriptorsetCapsule, (void *)ds);

  return Py_BuildValue("O", descriptorsetCapsule) // "O" for "Python object"
}

PyObject* delete_object(PyObject* self, PyObject* args)
{
  PyObject* descriptorsetCapsule;

  PyArg_ParseTuple(args, "O", &descriptorsetCapsule);

  symmetryFunctionSet* ds = (symmetryFunctionSet*)PyCapsule_GetPointer(
    descriptorsetCapsule, "DescribSetPtr");

  delete ds;

  // return nothing
  return Py_BuildValue("");
}

PyObject* print_symFuns(PyObject* self, PyObject* args)
{
  PyObject* descriptorsetCapsule;

  PyArg_ParseTuple(args, "O", &descriptorsetCapsule);

  symmetryFunctionSet* ds = (symmetryFunctionSet*)PyCapsule_GetPointer(
    descriptorsetCapsule, "DescribSetPtr");

  ds->print_symFuns();

  // return nothing
  return Py_BuildValue("");
}

PyMethodDef cDescriptorSetFunctions[] =
{
/*
 *  Structures which define functions ("methods") provided by the module.
 */
    {"construct",                   // C++/Py Constructor
      construct, METH_VARARGS,
     "Create `DescriptorSet` object"},

/*    {"fuel_up",                     // C++/Py wrapper for `fuel_up`
      fuel_up, METH_VARARGS,
     "Fuel up car"},

    {"drive",                       // C++/Py wrapper for `drive`
      drive, METH_VARARGS,
     "Drive the car"},

    {"print_mileage",               // C++/Py wrapper for `print_mileage`
      print_mileage, METH_VARARGS,
     "Print mileage of the car"},
*/
    {"delete_object",               // C++/Py Destructor
      delete_object, METH_VARARGS,
     "Delete `DescriptorSet` object"},

    {NULL, NULL, 0, NULL}      // Last function description must be empty.
                               // Otherwise, it will create seg fault while
                               // importing the module.
};


struct PyModuleDef cDescriptorSetModule =
{
/*
 *  Structure which defines the module.
 *
 *  For more info look at: https://docs.python.org/3/c-api/module.html
 *
 */
   PyModuleDef_HEAD_INIT,
   "cDescriptorSet",               // Name of the module.

   NULL,                 // Docstring for the module - in this case empty.

   -1,                   // Used by sub-interpreters, if you do not know what
                         // it is then you do not need it, keep -1 .

   cDescriptorSetFunctions  // Structures of type `PyMethodDef` with functions
                            // (or "methods") provided by the module.
};


PyMODINIT_FUNC PyInit_cDescriptorSet(void)
{
/*
 *   Function which initialises the Python module.
 *
 *   Note:  This function must be named "PyInit_MODULENAME",
 *          where "MODULENAME" is the name of the module.
 *
 */
    return PyModule_Create(&cDescriptorSetModule);
}
