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
                print(dGa[0])

                Gi_drp = sfs_cpp.eval(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri+dr]]))
                Gi_drm = sfs_cpp.eval(types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri-dr]]))
                dGn = [(Gi_drp[i] - Gi_drm[i])/(2*dr) for i in [0,1]]
                np.testing.assert_array_equal(dGn[0], dGn[1])

                np.testing.assert_array_almost_equal(dGa[0][:,-1], dGn[0])

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
            sfs_cpp.add_angular_functions([1.0], [1.0], etas)

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
