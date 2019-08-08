from os.path import dirname, abspath, join, normpath
from inspect import getsourcefile
from itertools import product, combinations_with_replacement
from warnings import warn
import numpy as _np
import ctypes as _ct

try:

    # TODO: the solution with relative path is really dirty.
    #    Better find a way to retrieve the main package's root path
    #    and use relative path from there.
    module_path = dirname(abspath(getsourcefile(lambda: 0)))
    lib = _ct.cdll.LoadLibrary(
        normpath(join(
            module_path,
            "libDescriptorSet.so")
        )
    )
    lib.create_descriptor_set.restype = _ct.c_void_p
    lib.create_descriptor_set.argtypes = (_ct.c_int,)
    lib.destroy_descriptor_set.argtypes = (_ct.c_void_p,)
    lib.descriptor_set_add_two_body_descriptor.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.POINTER(_ct.c_double), _ct.c_int, _ct.c_double)
    lib.descriptor_set_add_three_body_descriptor.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.POINTER(_ct.c_double), _ct.c_int, _ct.c_double)
    lib.descriptor_set_eval.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=2,
                                flags="C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=1,
                                flags="C_CONTIGUOUS"))
    lib.descriptor_set_eval_derivatives.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=2,
                                flags="C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=3,
                                flags="C_CONTIGUOUS"))
    lib.descriptor_set_eval_with_derivatives.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=2,
                                flags="C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=1,
                                flags="C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=3,
                                flags="C_CONTIGUOUS"))
    lib.get_cutoff_function_by_name.argtypes = (_ct.c_char_p,)
    lib.get_two_body_descriptor_by_name.argtypes = (_ct.c_char_p,)
    lib.get_three_body_descriptor_by_name.argtypes = (_ct.c_char_p,)
    lib.descriptor_set_get_G_vector_size.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int))
    lib.descriptor_set_print_descriptors.argtypes = (_ct.c_void_p,)
    lib.descriptor_set_eval_atomwise.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=2,
                                flags="C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=1,
                                flags="C_CONTIGUOUS"))
    lib.descriptor_set_eval_derivatives_atomwise.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=2,
                                flags="C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=3,
                                flags="C_CONTIGUOUS"))
    lib.descriptor_set_eval_with_derivatives_atomwise.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=2,
                                flags="C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=1,
                                flags="C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim=3,
                                flags="C_CONTIGUOUS"))
except OSError as e:
    # Possibly switch to a python based implementation if loading the dll fails
    raise e


class DescriptorSet(object):
    def __init__(self, atomtypes, cutoff=7.0):
        self.cutoff = cutoff
        self.atomtypes = atomtypes
        self.type_dict = {}
        self.num_Gs = [0]*len(atomtypes)
        for i, t in enumerate(atomtypes):
            self.type_dict[t] = i
            self.type_dict[i] = i
        self.obj = lib.create_descriptor_set(len(atomtypes))
        self._closed = False

    def close(self):
        lib.destroy_descriptor_set(self.obj)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def add_two_body_descriptor(self, type1, type2, funtype, prms,
                                cuttype="cos", cutoff=None):
        warn('The Behler atom-centered symmetry functions have been renamed ' +
             'and now follow the convention of J. Behler, JCP 134 074106')
        if cutoff is None:
            cutoff = self.cutoff
        cutid = lib.get_cutoff_function_by_name(cuttype.encode('utf-8'))
        if cutid == -1:
            raise TypeError("Unknown cutoff function type {}".format(cuttype))
        funid = lib.get_two_body_descriptor_by_name(funtype.encode('utf-8'))
        if funid == -1:
            raise TypeError("Unknown two body descriptor type: {}".format(
                funtype))
        ptr = (_ct.c_double*len(prms))(*prms)
        lib.descriptor_set_add_two_body_descriptor(
            self.obj, self.type_dict[type1], self.type_dict[type2], funid,
            len(prms), ptr, cutid, cutoff)
        self.num_Gs[self.type_dict[type1]] += 1

    def add_three_body_descriptor(self, type1, type2, type3, funtype, prms,
                                  cuttype="cos", cutoff=None):
        warn('The Behler atom-centered symmetry functions have been renamed ' +
             'and now follow the convention of J. Behler, JCP 134 074106')
        if cutoff is None:
            cutoff = self.cutoff
        cutid = lib.get_cutoff_function_by_name(cuttype.encode('utf-8'))
        if cutid == -1:
            raise TypeError("Unknown cutoff function type {}".format(cuttype))
        funid = lib.get_three_body_descriptor_by_name(funtype.encode('utf-8'))
        if funid == -1:
            raise TypeError(
                "Unknown three body descriptor type: {}".format(funtype))
        ptr = (_ct.c_double*len(prms))(*prms)
        lib.descriptor_set_add_three_body_descriptor(
            self.obj, self.type_dict[type1], self.type_dict[type2],
            self.type_dict[type3], funid, len(prms), ptr, cutid, cutoff)
        self.num_Gs[self.type_dict[type1]] += 1

    def add_G2_functions(self, rss, etas, cuttype="cos", cutoff=None):
        for rs, eta in zip(rss, etas):
            for (ti, tj) in product(self.atomtypes, repeat=2):
                self.add_two_body_descriptor(
                    ti, tj, "BehlerG2", [eta, rs], cuttype=cuttype,
                    cutoff=cutoff)

    def add_G2_functions_evenly(self, N):
        rss = _np.linspace(0., self.cutoff, N)
        etas = [2./(self.cutoff/(N-1))**2]*N
        self.add_radial_functions(rss, etas)

    def add_G5_functions(self, etas, zetas, lambs, cuttype="cos", cutoff=None):
        for eta in etas:
            for zeta in zetas:
                for lamb in lambs:
                    for ti in self.atomtypes:
                        for (tj, tk) in combinations_with_replacement(
                                self.atomtypes, 2):
                            self.add_three_body_descriptor(
                                ti, tj, tk, "BehlerG5", [lamb, zeta, eta],
                                cuttype=cuttype, cutoff=cutoff)

    def add_Artrith_Kolpak_set(self):
        # Parameters from Artrith and Kolpak Nano Lett. 2014, 14, 2670
        etas = [0.0009, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2]
        for t1 in self.atomtypes:
            for t2 in self.atomtypes:
                for eta in etas:
                    self.add_two_body_descriptor(
                        t1, t2, 'BehlerG2', [eta, 0.0], cuttype='cos',
                        cutoff=6.5)

        ang_etas = [0.0001, 0.003, 0.008]
        zetas = [1.0, 4.0]
        for ti in self.atomtypes:
            for (tj, tk) in combinations_with_replacement(
                    self.atomtypes, 2):
                for eta in ang_etas:
                    for zeta in zetas:
                        for lamb in [-1.0, 1.0]:
                            self.add_three_body_descriptor(
                                ti, tj, tk, "BehlerG4",
                                [lamb, zeta, eta], cuttype='cos', cutoff=6.5)

    def print_descriptors(self):
        lib.descriptor_set_print_descriptors(self.obj)

    def available_descriptors(self):
        lib.descriptor_set_available_descriptors(self.obj)

    def eval(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        # For each atom save how many symmetry functions are centered on it:
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        out = _np.zeros(sum(num_Gs_per_atom))
        lib.descriptor_set_eval(
            self.obj, len(types), types_ptr, xyzs, out)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return [out[cum_num_Gs[i]:cum_num_Gs[i+1]] for i in range(len(types))]

    def eval_derivatives(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        # For each atom save how many symmetry functions are centered on it:
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        out = _np.zeros((sum(num_Gs_per_atom), len(types), 3))
        lib.descriptor_set_eval_derivatives(
            self.obj, len(types), types_ptr, xyzs, out)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return [out[cum_num_Gs[i]:cum_num_Gs[i+1], :]
                for i in range(len(types))]

    def eval_with_derivatives(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        # For each atom save how many symmetry functions are centered on it:
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        Gs = _np.zeros(sum(num_Gs_per_atom))
        dGs = _np.zeros((sum(num_Gs_per_atom), len(types), 3))
        lib.descriptor_set_eval_with_derivatives(
            self.obj, len(types), types_ptr, xyzs, Gs, dGs)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return (
            [Gs[cum_num_Gs[i]:cum_num_Gs[i+1]] for i in range(len(types))],
            [dGs[cum_num_Gs[i]:cum_num_Gs[i+1], :] for i in range(len(types))])

    def eval_ase(self, atoms, derivatives=False):
        int_types = [self.type_dict[ti] for ti in atoms.get_chemical_symbols()]
        types_ptr = (_ct.c_int*len(atoms))(*int_types)
        # For each atom save how many symmetry functions are centered on it:
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        # The cummulative sum is used to determine the position of the symmetry
        # functions in the larger G vector
        cum_Gs = _np.cumsum([0]+num_Gs_per_atom)
        Gs = _np.zeros(sum(num_Gs_per_atom))
        if derivatives:
            dGs = _np.zeros((sum(num_Gs_per_atom), len(atoms), 3))
            lib.descriptor_set_eval_with_derivatives_atomwise(
                self.obj, len(atoms), types_ptr, atoms.get_positions(), Gs,
                dGs)
            return (
                [Gs[cum_Gs[i]:cum_Gs[i+1]] for i in range(len(atoms))],
                [dGs[cum_Gs[i]:cum_Gs[i+1], :] for i in range(len(atoms))])
        else:
            lib.descriptor_set_eval_atomwise(
                self.obj, len(atoms), types_ptr, atoms.get_positions(), Gs)
            return [Gs[cum_Gs[i]:cum_Gs[i+1]] for i in range(len(atoms))]

    def eval_geometry(self, geo):
        types = [a[0] for a in geo]
        xyzs = _np.array([a[1] for a in geo])
        return self.eval(types, xyzs)

    def eval_geometry_derivatives(self, geo):
        types = [a[0] for a in geo]
        xyzs = _np.array([a[1] for a in geo])
        return self.eval_derivatives(types, xyzs)

    def eval_atomwise(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        # len_G_vector = lib.SymmetryFunctionSet_get_G_vector_size(self.obj,
        #    len(types), types_ptr)
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        out = _np.zeros(sum(num_Gs_per_atom))
        lib.descriptor_set_eval_atomwise(
            self.obj, len(types), types_ptr, xyzs, out)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return [out[cum_num_Gs[i]:cum_num_Gs[i+1]] for i in range(len(types))]

    def eval_derivatives_atomwise(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        # len_G_vector = lib.SymmetryFunctionSet_get_G_vector_size(
        #    self.obj, len(types), types_ptr)
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        dGs = _np.zeros((sum(num_Gs_per_atom), len(types), 3))
        lib.descriptor_set_eval_derivatives_atomwise(
            self.obj, len(types), types_ptr, xyzs, dGs)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return [dGs[cum_num_Gs[i]:cum_num_Gs[i+1], :]
                for i in range(len(types))]

    def eval_with_derivatives_atomwise(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        # For each atom save how many symmetry functions are centered on it:
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        Gs = _np.zeros(sum(num_Gs_per_atom))
        dGs = _np.zeros((sum(num_Gs_per_atom), len(types), 3))
        lib.descriptor_set_eval_with_derivatives_atomwise(
            self.obj, len(types), types_ptr, xyzs, Gs, dGs)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return (
            [Gs[cum_num_Gs[i]:cum_num_Gs[i+1]] for i in range(len(types))],
            [dGs[cum_num_Gs[i]:cum_num_Gs[i+1], :] for i in range(len(types))])

    def eval_geometry_atomwise(self, geo):
        types = [a[0] for a in geo]
        xyzs = _np.array([a[1] for a in geo])
        return self.eval_atomwise(types, xyzs)
