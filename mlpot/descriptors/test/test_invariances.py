import unittest
from DescriptorLib.SymmetryFunctionSet import SymmetryFunctionSet
import numpy as np
from itertools import product, combinations_with_replacement

class LibraryTest(unittest.TestCase):
    def test_invariances(self):
        with SymmetryFunctionSet(['C', 'H'], cutoff = 6.5) as sfs:
            # Parameters from Artrith and Kolpak Nano Lett. 2014, 14, 2670
            etas = [0.0009, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2]
            for t1 in sfs.atomtypes:
                for t2 in sfs.atomtypes:
                    for eta in etas:
                        sfs.add_TwoBodySymmetryFunction(t1, t2, 'BehlerG1',
                            [eta], cuttype='cos')

            # Last two appended by Ralf Meyer
            ang_etas = [0.0001, 0.003, 0.008]#, 0.02, 0.1]
            zetas = [1.0, 4.0]
            for ti in sfs.atomtypes:
                for (tj, tk) in combinations_with_replacement(
                    sfs.atomtypes, 2):
                        for eta in ang_etas:
                            for lamb in [-1.0, 1.0]:
                                for zeta in zetas:
                                    sfs.add_ThreeBodySymmetryFunction(
                                        ti, tj, tk, 'BehlerG3', [lamb, zeta, eta],
                                        cuttype = 'cos')

            types = ['H', 'H', 'H', 'C', 'C', 'H', 'H', 'H']
            xyzs = np.array([[1.0217062478, 0.0000000000, 1.1651331805],
                            [-0.5108531239, 0.8848235658, 1.1651331805],
                            [-0.5108531239, -0.8848235658, 1.1651331805],
                            [0.0000000000, 0.0000000000,  0.7662728375],
                            [0.0000000000, 0.0000000000, -0.7662728375],
                            [0.5108531239, -0.8848235658, -1.1651331805],
                            [-1.0217062478, 0.0000000000, -1.1651331805],
                            [0.5108531239, 0.8848235658, -1.1651331805]])

            Gs, dGs = sfs.eval_with_derivatives_atomwise(types, xyzs)
            Gs = np.asarray(Gs)
            dGs = np.asarray(dGs)

            # Test for translational invariance
            Gs_test, dGs_test = sfs.eval_with_derivatives_atomwise(
                types, xyzs + 10.)
            Gs_test = np.asarray(Gs_test)
            dGs_test = np.asarray(dGs_test)
            np.testing.assert_allclose(Gs, Gs_test, atol = 1E-7)
            np.testing.assert_allclose(dGs, dGs_test, atol = 1E-7)

            # Test for rotational invariance
            def rotation_matrix(axis, theta):
                """
                Stolen from https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
                Return the rotation matrix associated with counterclockwise
                rotation about the given axis by theta radians.
                """
                axis = np.asarray(axis)
                axis = axis / np.linalg.norm(axis)
                a = np.cos(theta / 2.0)
                b, c, d = -axis * np.sin(theta / 2.0)
                aa, bb, cc, dd = a * a, b * b, c * c, d * d
                bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
                return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                                 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                                 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

            R = rotation_matrix([2,1,3], np.pi/3.)
            Gs_test, dGs_test = sfs.eval_with_derivatives_atomwise(
                types, xyzs.dot(R))
            Gs_test = np.asarray(Gs_test)
            # dGs_test has to be rotated back to the original orientation
            dGs_test = np.asarray(dGs_test).dot(np.linalg.inv(R))
            np.testing.assert_allclose(Gs, Gs_test, atol = 1E-7)
            np.testing.assert_allclose(dGs, dGs_test,
                atol = 1E-7)

            # Test for rotational invariance
            xyzs = xyzs[[1,0,2,4,3,6,5,7]]
            Gs_test, dGs_test = sfs.eval_with_derivatives_atomwise(
                types, xyzs + 10.)
            Gs_test = np.asarray(Gs_test)
            dGs_test = np.asarray(dGs_test)
            np.testing.assert_allclose(Gs, Gs_test, atol = 1E-7)
            # TODO: finish test for permutational invariance by figuring out
            # the inverse transformation for dGs_test array
            #np.testing.assert_allclose(dGs, dGs_test, atol = 1E-7)


if __name__ == '__main__':
    unittest.main()
