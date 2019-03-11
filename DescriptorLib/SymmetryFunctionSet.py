from os.path import dirname, abspath, join, normpath
from inspect import getsourcefile
from itertools import product, combinations_with_replacement
from DescriptorLib import SymmetryFunctions as SFs
import numpy as _np
import ctypes as _ct
from scipy.spatial.distance import pdist, squareform
from scipy.misc import comb

try:

    # TODO: the solution with relative path is really dirty.
    #    Better find a way to retrieve the main package's root path
    #    and use relative path from there.
    module_path = dirname(abspath(getsourcefile(lambda:0)))
    lib = _ct.cdll.LoadLibrary(
        normpath(join(
            module_path,
            "libSymFunSet.so")
        )
    )
    lib.create_SymmetryFunctionSet.restype = _ct.c_void_p
    lib.create_SymmetryFunctionSet.argtypes = (_ct.c_int,)
    lib.destroy_SymmetryFunctionSet.argtypes = (_ct.c_void_p,)
    lib.SymmetryFunctionSet_add_TwoBodySymmetryFunction.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.POINTER(_ct.c_double), _ct.c_int, _ct.c_double)
    lib.SymmetryFunctionSet_add_ThreeBodySymmetryFunction.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int, _ct.c_int,
        _ct.POINTER(_ct.c_double), _ct.c_int, _ct.c_double)
    lib.SymmetryFunctionSet_eval.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 1, flags = "C_CONTIGUOUS"))
    lib.SymmetryFunctionSet_eval_derivatives.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 3, flags = "C_CONTIGUOUS"))
    lib.SymmetryFunctionSet_eval_with_derivatives.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 1, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 3, flags = "C_CONTIGUOUS"))
    lib.SymmetryFunctionSet_get_CutFun_by_name.argtypes = (_ct.c_char_p,)
    lib.SymmetryFunctionSet_get_TwoBodySymFun_by_name.argtypes = (_ct.c_char_p,)
    lib.SymmetryFunctionSet_get_ThreeBodySymFun_by_name.argtypes = (
        _ct.c_char_p,)
    lib.SymmetryFunctionSet_get_G_vector_size.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int))
    lib.SymmetryFunctionSet_print_symFuns.argtypes = (_ct.c_void_p,)
    lib.SymmetryFunctionSet_eval_atomwise.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 1, flags = "C_CONTIGUOUS"))
    lib.SymmetryFunctionSet_eval_derivatives_atomwise.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 3, flags = "C_CONTIGUOUS"))
    lib.SymmetryFunctionSet_eval_with_derivatives_atomwise.argtypes = (
        _ct.c_void_p, _ct.c_int, _ct.POINTER(_ct.c_int),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 2, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 1, flags = "C_CONTIGUOUS"),
        _np.ctypeslib.ndpointer(dtype=_np.float64, ndim = 3, flags = "C_CONTIGUOUS"))
except OSError as e:
    # Possibly switch to a python based implementation if loading the dll fails
    raise OSError(e.message)

class SymmetryFunctionSet(object):
    def __init__(self, atomtypes, cutoff = 7.0):
        self.cutoff = cutoff
        self.atomtypes = atomtypes
        self.type_dict = {}
        self.num_Gs = [0]*len(atomtypes)
        for i, t in enumerate(atomtypes):
            self.type_dict[t] = i
            self.type_dict[i] = i
        self.obj = lib.create_SymmetryFunctionSet(len(atomtypes))
        self._closed = False

    def close(self):
        lib.destroy_SymmetryFunctionSet(self.obj)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self ,type, value, traceback):
        self.close()

    def add_TwoBodySymmetryFunction(self, type1, type2, funtype, prms,
            cuttype = "cos", cutoff = None):
        if cutoff == None:
            cutoff = self.cutoff
        cutid = lib.SymmetryFunctionSet_get_CutFun_by_name(
            cuttype.encode('utf-8'))
        if cutid == -1:
            raise TypeError("Unknow CutoffFunction type {}".format(cuttype))
        funid = lib.SymmetryFunctionSet_get_TwoBodySymFun_by_name(
            funtype.encode('utf-8'))
        if funid == -1:
            raise TypeError("Unknown TwoBodySymmetryFunction type: {}".format(
                funtype))
        ptr = (_ct.c_double*len(prms))(*prms)
        lib.SymmetryFunctionSet_add_TwoBodySymmetryFunction(self.obj,
            self.type_dict[type1], self.type_dict[type2], funid, len(prms),
            ptr, cutid, cutoff)
        self.num_Gs[self.type_dict[type1]] += 1

    def add_ThreeBodySymmetryFunction(self, type1, type2, type3, funtype, prms,
            cuttype = "cos", cutoff = None):
        if cutoff == None:
            cutoff = self.cutoff
        cutid = lib.SymmetryFunctionSet_get_CutFun_by_name(
            cuttype.encode('utf-8'))
        if cutid == -1:
            raise TypeError("Unknow CutoffFunction type {}".format(cuttype))
        funid = lib.SymmetryFunctionSet_get_ThreeBodySymFun_by_name(
            funtype.encode('utf-8'))
        if funid == -1:
            raise TypeError("Unknown ThreeBodySymmetryFunction type: {}".format(
                funtype))
        ptr = (_ct.c_double*len(prms))(*prms)
        lib.SymmetryFunctionSet_add_ThreeBodySymmetryFunction(self.obj,
            self.type_dict[type1], self.type_dict[type2], self.type_dict[type3],
            funid, len(prms), ptr, cutid, cutoff)
        self.num_Gs[self.type_dict[type1]] += 1

    def add_radial_functions(self, rss, etas, cuttype = "cos", cutoff = None):
        for rs, eta in zip(rss, etas):
            for (ti, tj) in product(self.atomtypes, repeat = 2):
                self.add_TwoBodySymmetryFunction(ti, tj, "BehlerG2", [eta, rs],
                    cuttype = cuttype, cutoff = cutoff)

    def add_radial_functions_evenly(self, N):
        rss = _np.linspace(0.,self.cutoff,N)
        etas = [2./(self.cutoff/(N-1))**2]*N
        self.add_radial_functions(rss, etas)

    def add_angular_functions(self, etas, zetas, lambs, cuttype = "cos",
            cutoff = None):
        for eta in etas:
            for zeta in zetas:
                for lamb in lambs:
                    for ti in self.atomtypes:
                        for (tj, tk) in combinations_with_replacement(
                                self.atomtypes, 2):
                            self.add_ThreeBodySymmetryFunction(ti, tj, tk,
                                "BehlerG4", [lamb, zeta, eta],
                                cuttype = cuttype, cutoff = cutoff)

    def print_symFuns(self):
        lib.SymmetryFunctionSet_print_symFuns(self.obj)

    def available_symFuns(self):
        lib.SymmetryFunctionSet_available_symFuns(self.obj)

    def eval(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        # For each atom save how many symmetry functions are centered on it:
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        out = _np.zeros(sum(num_Gs_per_atom))
        lib.SymmetryFunctionSet_eval(self.obj, len(types), types_ptr, xyzs, out)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return [out[cum_num_Gs[i]:cum_num_Gs[i+1]] for i in range(len(types))]

    def eval_derivatives(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        # For each atom save how many symmetry functions are centered on it:
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        out = _np.zeros((sum(num_Gs_per_atom), len(types), 3))
        lib.SymmetryFunctionSet_eval_derivatives(
            self.obj, len(types), types_ptr, xyzs, out)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return [out[cum_num_Gs[i]:cum_num_Gs[i+1],:] for i in range(len(types))]

    def eval_with_derivatives(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        # For each atom save how many symmetry functions are centered on it:
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        Gs = _np.zeros(sum(num_Gs_per_atom))
        dGs = _np.zeros((sum(num_Gs_per_atom), len(types), 3))
        lib.SymmetryFunctionSet_eval_with_derivatives(
            self.obj, len(types), types_ptr, xyzs, Gs, dGs)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return ([Gs[cum_num_Gs[i]:cum_num_Gs[i+1]] for i in range(len(types))],
            [dGs[cum_num_Gs[i]:cum_num_Gs[i+1],:] for i in range(len(types))])

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
            lib.SymmetryFunctionSet_eval_with_derivatives_atomwise(
                self.obj, len(atoms), types_ptr, atoms.get_positions(), Gs, dGs)
            return ([Gs[cum_Gs[i]:cum_Gs[i+1]] for i in range(len(atoms))],
                [dGs[cum_Gs[i]:cum_Gs[i+1],:] for i in range(len(atoms))])
        else:
            lib.SymmetryFunctionSet_eval_atomwise(
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
        #len_G_vector = lib.SymmetryFunctionSet_get_G_vector_size(self.obj,
        #    len(types), types_ptr)
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        out = _np.zeros(sum(num_Gs_per_atom))
        lib.SymmetryFunctionSet_eval_atomwise(
            self.obj, len(types), types_ptr, xyzs, out)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return [out[cum_num_Gs[i]:cum_num_Gs[i+1]] for i in range(len(types))]

    def eval_derivatives_atomwise(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        #len_G_vector = lib.SymmetryFunctionSet_get_G_vector_size(
        #    self.obj, len(types), types_ptr)
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        dGs = _np.zeros((sum(num_Gs_per_atom), len(types), 3))
        lib.SymmetryFunctionSet_eval_derivatives_atomwise(
            self.obj, len(types), types_ptr, xyzs, dGs)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return [dGs[cum_num_Gs[i]:cum_num_Gs[i+1],:] for i in range(len(types))]

    def eval_with_derivatives_atomwise(self, types, xyzs):
        int_types = [self.type_dict[ti] for ti in types]
        types_ptr = (_ct.c_int*len(types))(*int_types)
        # For each atom save how many symmetry functions are centered on it:
        num_Gs_per_atom = [self.num_Gs[ti] for ti in int_types]
        Gs = _np.zeros(sum(num_Gs_per_atom))
        dGs = _np.zeros((sum(num_Gs_per_atom), len(types), 3))
        lib.SymmetryFunctionSet_eval_with_derivatives_atomwise(
            self.obj, len(types), types_ptr, xyzs, Gs, dGs)
        cum_num_Gs = _np.cumsum([0]+num_Gs_per_atom)
        return ([Gs[cum_num_Gs[i]:cum_num_Gs[i+1]] for i in range(len(types))],
            [dGs[cum_num_Gs[i]:cum_num_Gs[i+1],:] for i in range(len(types))])

    def eval_geometry_atomwise(self, geo):
        types = [a[0] for a in geo]
        xyzs = _np.array([a[1] for a in geo])
        return self.eval_atomwise(types, xyzs)

class SymmetryFunctionSet_py(object):

    def __init__(self, atomtypes, cutoff = 7.):
        self.atomtypes = atomtypes
        self.cutoff = cutoff
        self.radial_sym_funs = []
        self.angular_sym_funs = []

    def add_radial_functions(self, rss, etas):
        for rs in rss:
            for eta in etas:
                self.radial_sym_funs.append(
                        SFs.RadialSymmetryFunction(rs, eta, self.cutoff))

    def add_angular_functions(self, etas, zetas, lambs):
        for eta in etas:
            for zeta in zetas:
                for lamb in lambs:
                    self.angular_sym_funs.append(
                        SFs.AngularSymmetryFunction(eta, zeta, lamb, self.cutoff))

    def add_angular_functions_new(self, etas, zetas, lambs, rss):
        for eta in etas:
            for zeta in zetas:
                for lamb in lambs:
                    for rs in rss:
                        self.angular_sym_funs.append(
                            SFs.AngularSymmetryFunction(eta, zeta, lamb, rs, self.cutoff))

    def add_radial_functions_evenly(self, N):
        rss = _np.linspace(0.,self.cutoff,N)
        etas = [2./(self.cutoff/(N-1))**2]*N
        for rs, eta in zip(rss, etas):
            self.radial_sym_funs.append(
                        SFs.RadialSymmetryFunction(rs, eta, self.cutoff))

    def eval_geometry(self, geometry, derivative = False):
        # Returns a (Number of atoms) x (Size of G vector) matrix
        # The G vector doubles in size if derivatives are also requested
        # Calculate distance matrix. Should be solvable without using
        # squareform!
        # TODO: rewrite even more efficient
        # TODO: Implement derivative
        N = len(geometry) # Number of atoms
        Nt = len(self.atomtypes) # Number of atomtypes
        Nr = len(self.radial_sym_funs) # Number of radial symmetry functions
        Na = len(self.angular_sym_funs) # Number of angular symmetry functions

        dist_mat = squareform(pdist([g[1] for g in geometry]))
        # Needed for angular symmetry functions
        # maybe more elegant solution possible using transposition?
        rij = _np.tile(dist_mat.reshape((N,N,1)),(1,1,N))
        rik = _np.tile(dist_mat.reshape((N,1,N)),(1,N,1))
        rjk = _np.tile(dist_mat.reshape((1,N,N)),(N,1,1))
        costheta = (rij**2+rik**2-rjk**2)
        costheta[rij*rik > 0] = costheta[rij*rik > 0] / ((2*rij*rik)[rij*rik > 0])
        costheta[rij*rik == 0] = 0.0
        # (1-eye) to satify the j != i condition of the sum
        kron_ij = (1.-_np.eye(N))
        # Similar for the condition j != i, k != j in the angular sum
        dij = _np.tile(_np.eye(N).reshape((N,N,1)),(1,1,N))
        dik = _np.tile(_np.eye(N).reshape((N,1,N)),(1,N,1))
        djk = _np.tile(_np.eye(N).reshape((1,N,N)),(N,1,1))
        kron_ijk = 1. - _np.sign(dij+dik+djk)

        if derivative == False:
            out = _np.zeros((N, Nr*Nt + comb(Nt, 2, exact = True,
                                             repetition = True)*Na))

            ind = 0 # Counter for the combinations of angle types
            for t, atype in enumerate(self.atomtypes):
                # Mask for the different atom types
                mask = [a[0] == atype for a in geometry]
                for i, rad_fun in enumerate(self.radial_sym_funs):
                    out[:,t*Nr+i] = (kron_ij * rad_fun(dist_mat)).dot(mask)
                for atype2 in self.atomtypes[t:]:
                    # Second mask because two atom types are involved
                    mask2 = [a[0] == atype2 for a in geometry]
                    for j, ang_fun in enumerate(self.angular_sym_funs):
                        if (atype == atype2):
                            out[:,Nt*Nr+ind*Na+j] = 0.5*(kron_ijk *
                                ang_fun(rij, rik, costheta)).dot(mask).dot(mask2)
                        else:
                            out[:,Nt*Nr+ind*Na+j] = (kron_ijk *
                                ang_fun(rij, rik, costheta)).dot(mask).dot(mask2)
                    ind += 1

        else: # derivative = True: doubles the size of the output matrix
            out = _np.zeros((N, 2*(Nr*Nt + comb(Nt, 2, exact = True,
                                             repetition = True)*Na)))

            ind = 0 # Counter for the combinations of angle types
            for t, atype in enumerate(self.atomtypes):
                # Mask for the different atom types
                mask = [a[0] == atype for a in geometry]
                for i, rad_fun in enumerate(self.radial_sym_funs):
                    out[:,t*2*Nr+2*i] = (kron_ij * rad_fun(dist_mat)).dot(mask)
                    out[:,t*2*Nr+2*i+1] = (kron_ij *
                                        rad_fun.derivative(dist_mat)).dot(mask)
                for atype2 in self.atomtypes[t:]:
                    # Second mask because two atom types are involved
                    mask2 = [a[0] == atype2 for a in geometry]
                    for j, ang_fun in enumerate(self.angular_sym_funs):
                        out[:,Nt*2*Nr+ind*2*Na+2*j] = (kron_ijk *
                            ang_fun(rij, rik, costheta)).dot(mask).dot(mask2)
                        out[:,Nt*2*Nr+ind*2*Na+2*j+1] = (kron_ijk *
                            ang_fun.derivative(rij, rik, costheta)).dot(mask).dot(mask2)
                    ind += 1
        return out
