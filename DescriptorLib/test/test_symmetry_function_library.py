import unittest
from DescriptorLib.SymmetryFunctionSet import SymmetryFunctionSet as SymFunSet_cpp
import numpy as np
from itertools import product

class LibraryTest(unittest.TestCase):

    def test_cos_cutoff_functions(self):
        r_vec = np.linspace(0.1,7,101)
        geos = [[("F", np.array([0.0, 0.0, 0.0])),
                ("H", np.array([0.0, 0.0, ri]))] for ri in r_vec]
        with SymFunSet_cpp(["H", "F"], cutoff = 6.5) as sfs:
            for (t1, t2) in product(sfs.atomtypes, repeat = 2):
                sfs.add_TwoBodySymmetryFunction(
                    t1, t2, "BehlerG0", [], cuttype = "cos")

            Gs = []
            dGs = []
            for geo in geos:
                Gs.append(sfs.eval_geometry(geo))
                dGs.append(sfs.eval_geometry_derivatives(geo))
            np.testing.assert_allclose(np.array(Gs)[:,0,0],
                0.5*(1.0+np.cos(r_vec*np.pi/sfs.cutoff))*(r_vec < sfs.cutoff))
            np.testing.assert_allclose(np.array(dGs)[:,0,0,-1],
                0.5*(-np.sin(r_vec*np.pi/sfs.cutoff)*np.pi/sfs.cutoff)*(
                r_vec < sfs.cutoff))

    def test_poly_cutoff_functions(self):
        r_vec = np.linspace(0.1,7,101)
        geos = [[("F", np.array([0.0, 0.0, 0.0])),
                ("H", np.array([0.0, 0.0, ri]))] for ri in r_vec]
        with SymFunSet_cpp(["H", "F"], cutoff = 6.5) as sfs:
            for (t1, t2) in product(sfs.atomtypes, repeat = 2):
                sfs.add_TwoBodySymmetryFunction(
                    t1, t2, "BehlerG0", [], cuttype = "polynomial")

            Gs = []
            dGs = []
            for geo in geos:
                Gs.append(sfs.eval_geometry(geo))
                dGs.append(sfs.eval_geometry_derivatives(geo))
            np.testing.assert_allclose(np.array(Gs)[:,0,0],
                (1 - 10.0 * (r_vec/sfs.cutoff)**3
                 + 15.0 * (r_vec/sfs.cutoff)**4
                - 6.0 * (r_vec/sfs.cutoff)**5)*(r_vec < sfs.cutoff))
            np.testing.assert_allclose(np.array(dGs)[:,0,0,-1],
                (- 30.0 * (r_vec/sfs.cutoff)**2/sfs.cutoff
                 + 60.0 * (r_vec/sfs.cutoff)**3/sfs.cutoff
                - 30.0 * (r_vec/sfs.cutoff)**4/sfs.cutoff)*(r_vec < sfs.cutoff))

    def test_tanh_cutoff_functions(self):
        r_vec = np.linspace(0.1,7,101)
        geos = [[("F", np.array([0.0, 0.0, 0.0])),
                ("H", np.array([0.0, 0.0, ri]))] for ri in r_vec]
        with SymFunSet_cpp(["H", "F"], cutoff = 6.5) as sfs:
            for (t1, t2) in product(sfs.atomtypes, repeat = 2):
                sfs.add_TwoBodySymmetryFunction(
                    t1, t2, "BehlerG0", [], cuttype = "tanh")

            Gs = []
            dGs = []
            for geo in geos:
                Gs.append(sfs.eval_geometry(geo))
                dGs.append(sfs.eval_geometry_derivatives(geo))
            np.testing.assert_allclose(np.array(Gs)[:,0,0],
                (np.tanh(1.0-r_vec/sfs.cutoff)**3)*(r_vec < sfs.cutoff))
            np.testing.assert_allclose(np.array(dGs)[:,0,0,-1],
                -(3*np.sinh(1.0-r_vec/sfs.cutoff)**2)/
                (sfs.cutoff*np.cosh(1.0-r_vec/sfs.cutoff)**4)*(
                r_vec < sfs.cutoff))

    def test_const_cutoff_functions(self):
        r_vec = np.linspace(0.1,7,101)
        geos = [[("F", np.array([0.0, 0.0, 0.0])),
                ("H", np.array([0.0, 0.0, ri]))] for ri in r_vec]
        with SymFunSet_cpp(["H", "F"], cutoff = 6.5) as sfs:
            for (t1, t2) in product(sfs.atomtypes, repeat = 2):
                sfs.add_TwoBodySymmetryFunction(
                    t1, t2, "BehlerG0", [], cuttype = "const")

            Gs = []
            dGs = []
            for geo in geos:
                Gs.append(sfs.eval_geometry(geo))
                dGs.append(sfs.eval_geometry_derivatives(geo))
            np.testing.assert_allclose(np.array(Gs)[:,0,0],
                (1.0)*(r_vec < sfs.cutoff))
            np.testing.assert_allclose(np.array(dGs)[:,0,0,-1], 0.0)

    def test_dimer_cos(self):
        with SymFunSet_cpp(["Au"], cutoff = 7.) as sfs_cpp:
            types = ["Au", "Au"]
            rss = [0.0, 0.0, 0.0]
            etas = np.array([0.01, 0.1, 1.0])

            sfs_cpp.add_radial_functions(rss, etas)

            dr = np.sqrt(np.finfo(float).eps)
            for ri in np.linspace(0.1,7,101):
                Gi = sfs_cpp.eval(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri]]))
                # Assert Symmmetry
                np.testing.assert_array_equal(Gi[0], Gi[1])
                # Assert Values
                np.testing.assert_allclose(
                    Gi[0], np.exp(-etas*(ri-rss)**2)*0.5*(1.0+np.cos(np.pi*ri/sfs_cpp.cutoff)))
                # Derivatives
                dGa = sfs_cpp.eval_derivatives(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri]]))
                # Assert Symmetry
                np.testing.assert_array_equal(dGa[0], dGa[1])
                # Assert Values
                np.testing.assert_allclose(dGa[0][:,-1], np.exp(-etas*(ri-rss)**2)*(
                    0.5*(1.0+np.cos(np.pi*ri/sfs_cpp.cutoff))*2.0*(-etas)*(ri-rss)+
                    0.5*(-np.sin(np.pi*ri/sfs_cpp.cutoff)*np.pi/sfs_cpp.cutoff)))

                Gi_drp = sfs_cpp.eval(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri+dr]]))
                Gi_drm = sfs_cpp.eval(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri-dr]]))
                dGn = [(Gi_drp[i] - Gi_drm[i])/(2*dr) for i in [0,1]]
                # Assert Symmetry
                np.testing.assert_array_equal(dGn[0], dGn[1])
                # Assert Values
                np.testing.assert_allclose(dGa[0][:,-1], dGn[0],
                    rtol = 1E-7, atol = 1E-7)

    def test_dimer_polynomial(self):
        with SymFunSet_cpp(["Au"], cutoff = 7.) as sfs_cpp:
            types = ["Au", "Au"]
            rss = [0.0, 0.0, 0.0]
            bohr2ang = 0.529177249
            etas = np.array([0.01, 0.1, 1.0])/bohr2ang**2

            sfs_cpp.add_radial_functions(rss, etas, cuttype = "polynomial")

            dr = np.sqrt(np.finfo(float).eps)
            for ri in np.linspace(2,7,10):
                Gi = sfs_cpp.eval(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri]]))
                # Assert Symmmetry
                np.testing.assert_array_equal(Gi[0], Gi[1])
                # Assert Values
                np.testing.assert_allclose(
                    Gi[0], np.exp(-etas*(ri-rss)**2)*(1 - 10.0 * (ri/sfs_cpp.cutoff)**3 +
                    15.0 * (ri/sfs_cpp.cutoff)**4 - 6.0 * (ri/sfs_cpp.cutoff)**5))
                # Derivatives
                dGa = sfs_cpp.eval_derivatives(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri]]))
                # Assert Symmmetry
                np.testing.assert_array_equal(dGa[0], dGa[1])
                # Assert Values
                np.testing.assert_allclose(np.exp(-etas*(ri-rss)**2)*(
                    (1 - 10.0 * (ri/sfs_cpp.cutoff)**3 + 15.0 * (ri/sfs_cpp.cutoff)**4
                    - 6.0 * (ri/sfs_cpp.cutoff)**5)*2.0*(-etas)*(ri-rss)
                    +(-30.0 * (ri**2/sfs_cpp.cutoff**3) + 60.0 * (ri**3/sfs_cpp.cutoff**4)
                    - 30.0 * (ri**4/sfs_cpp.cutoff**5))),
                    dGa[0][:,-1])

                Gi_drp = sfs_cpp.eval(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri+dr]]))
                Gi_drm = sfs_cpp.eval(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri-dr]]))
                dGn = [(Gi_drp[i] - Gi_drm[i])/(2*dr) for i in [0,1]]
                # Assert Symmetry
                np.testing.assert_array_equal(dGn[0], dGn[1])
                # Assert Values
                np.testing.assert_allclose(dGa[0][:,-1], dGn[0],
                    rtol = 1E-7, atol = 1E-7)

    def test_acetone(self):
        from scipy.optimize import approx_fprime
        x0 = np.array([0.00000,        0.00000,        0.00000, #C
                       1.40704,        0.00902,       -0.67203, #C
                       1.67062,       -0.92069,       -1.22124, #H
                       2.20762,        0.06960,        0.11291, #H
                       1.61784,        0.88539,       -1.32030, #H
                      -1.40732,       -0.00378,       -0.67926, #C
                      -1.65709,        0.91221,       -1.25741, #H
                      -2.20522,       -0.03081,        0.10912, #H
                      -1.64457,       -0.88332,       -1.31507, #H
                       0.00000,       -0.00000,        1.20367]) #O
        types = ["C", "C", "H", "H", "H", "C", "H", "H", "H", "O"]

        with SymFunSet_cpp(["C", "H", "O"], cutoff = 7.0) as sfs:
            radial_etas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
            rss = [0.0]*len(radial_etas)

            angular_etas = [0.0001, 0.003, 0.008]
            lambs = [1.0, -1.0]
            zetas = [1.0, 4.0]

            sfs.add_radial_functions(rss, radial_etas)
            sfs.add_angular_functions(angular_etas, zetas, lambs)
            f0 = sfs.eval(types, x0.reshape((-1,3)))
            eps = np.sqrt(np.finfo(float).eps)

            for i in range(len(f0)):
                for j in range(len(f0[i])):
                    def f(x):
                        return np.array(sfs.eval(types, x.reshape((-1,3))))[i][j]

                    np.testing.assert_array_almost_equal(
                        sfs.eval_derivatives(types, x0.reshape((-1,3)))[i][j],
                        approx_fprime(x0, f, epsilon = eps))


    def test_derivaties(self):
        with SymFunSet_cpp(["Ni", "Au"], cutoff = 10.) as sfs_cpp:
            pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0,-1.0, 0.0]])
            types = ["Ni", "Ni", "Au", "Ni", "Au"]

            rss = [0.0, 0.0, 0.0]
            etas = [1.0, 0.01, 0.0001]

            sfs_cpp.add_radial_functions(rss, etas)
            sfs_cpp.add_angular_functions(etas, [1.0], [1.0])

            out_cpp = sfs_cpp.eval(types, pos)
            analytical_derivatives = sfs_cpp.eval_derivatives(types, pos)
            numerical_derivatives = np.zeros((len(out_cpp), out_cpp[0].size, pos.size))
            dx = np.sqrt(np.finfo(float).eps)
            for i in range(pos.size):
                dpos = np.zeros(pos.shape)
                dpos[np.unravel_index(i,dpos.shape)] += dx
                numerical_derivatives[:,:,i] = (np.array(sfs_cpp.eval(types, pos+dpos))
                                 - np.array(sfs_cpp.eval(types, pos-dpos)))/(2*dx)

        np.testing.assert_array_almost_equal(numerical_derivatives.flatten(), np.array(analytical_derivatives).flatten())

if __name__ == '__main__':
    unittest.main()
