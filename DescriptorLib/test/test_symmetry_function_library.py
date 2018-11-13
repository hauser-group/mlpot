import unittest
from DescriptorLib.SymmetryFunctionSet import SymmetryFunctionSet as SymFunSet_cpp
import numpy as np

class LibraryTest(unittest.TestCase):

    def test_dimer(self):
        with SymFunSet_cpp(["Au"], cutoff = 7.) as sfs_cpp:
            types = ["Au", "Au"]
            rss = [0.0, 0.0, 0.0]
            etas = np.array([0.01, 0.1, 1.0])

            sfs_cpp.add_radial_functions(rss, etas)

            dr = 0.00001
            for ri in np.linspace(2,7,10):
                Gi = sfs_cpp.eval(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri]]))
                # Assert Symmmetry
                np.testing.assert_array_equal(Gi[0], Gi[1])
                # Assert Values
                np.testing.assert_array_equal(
                    Gi[0], np.exp(-etas*(ri-rss)**2)*0.5*(1.0+np.cos(np.pi*ri/sfs_cpp.cutoff)))
                # Derivatives
                dGa = sfs_cpp.eval_derivatives(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri]]))
                np.testing.assert_array_equal(dGa[0], dGa[1])

                Gi_drp = sfs_cpp.eval(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri+dr]]))
                Gi_drm = sfs_cpp.eval(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri-dr]]))
                dGn = [(Gi_drp[i] - Gi_drm[i])/(2*dr) for i in [0,1]]
                np.testing.assert_array_equal(dGn[0], dGn[1])

                np.testing.assert_array_almost_equal(dGa[0][:,-1], dGn[0])

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
        
        with SymFunSet_cpp(["C", "H", "O"]) as sfs:
            radial_etas = [0.0009, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2]
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
                
                    np.testing.assert_allclose(approx_fprime(x0, f, epsilon = eps),
                        sfs.eval_derivatives(types, x0.reshape((-1,3)))[i][j], 
                        rtol=1e-4, atol=1)
             

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
            dx = 0.00001
            for i in xrange(pos.size):
                dpos = np.zeros(pos.shape)
                dpos[np.unravel_index(i,dpos.shape)] += dx
                numerical_derivatives[:,:,i] = (np.array(sfs_cpp.eval(types, pos+dpos))
                                 - np.array(sfs_cpp.eval(types, pos-dpos)))/(2*dx)

        np.testing.assert_array_almost_equal(numerical_derivatives.flatten(), np.array(analytical_derivatives).flatten())

if __name__ == '__main__':
    unittest.main()
