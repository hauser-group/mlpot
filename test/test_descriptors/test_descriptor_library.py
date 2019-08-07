import unittest
from mlpot.descriptors import DescriptorSet
import numpy as np


class LibraryTest(unittest.TestCase):

    def test_print_functions(self):
        with DescriptorSet(['C', 'H', 'O']) as ds:
            ds.available_descriptors()
            ds.add_Artrith_Kolpak_set()
            ds.print_descriptors()

    def test_exceptions(self):
        with DescriptorSet(['H', 'O']) as ds:
            try:
                ds.add_two_body_descriptor('H', 'O', 'FOO', [])
            except TypeError:
                pass
            try:
                ds.add_three_body_descriptor('H', 'O', 'H', 'FOO', [])
            except TypeError:
                pass
            try:
                ds.add_two_body_descriptor('H', 'O', 'BehlerG1', [],
                                           cuttype='FOO')
            except TypeError:
                pass
            try:
                ds.add_three_body_descriptor('H', 'O', 'O', 'BehlerG4',
                                             [1.0, 1.0, 1.0], cuttype='FOO')
            except TypeError:
                pass

    def test_dimer_cos(self):
        with DescriptorSet(['Au'], cutoff=7.) as ds:
            types = ['Au', 'Au']
            rss = [0.0, 0.0, 0.0]
            etas = np.array([0.01, 0.1, 1.0])

            ds.add_G2_functions(rss, etas)

            def cutfun(r):
                return 0.5*(
                    1.0+np.cos(np.pi*r/ds.cutoff))*(r < ds.cutoff)

            dr = np.sqrt(np.finfo(float).eps)
            for ri in np.linspace(0.1, 8, 101):
                Gi = ds.eval(
                    types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri]]))
                # Assert Symmmetry
                np.testing.assert_array_equal(Gi[0], Gi[1])
                # Assert Values
                np.testing.assert_allclose(
                    Gi[0], np.exp(-etas*(ri-rss)**2)*cutfun(ri))
                # Derivatives
                dGa = ds.eval_derivatives(
                    types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri]]))
                # Assert Symmetry
                np.testing.assert_array_equal(dGa[0], dGa[1])
                # Assert Values
                np.testing.assert_allclose(
                    dGa[0][:, 1, -1],
                    np.exp(-etas*(ri-rss)**2)*(
                        cutfun(ri)*2.0*(-etas)*(ri-rss)
                        + 0.5*(-np.sin(np.pi*ri/ds.cutoff)
                               * np.pi/ds.cutoff)*(ri < ds.cutoff)))

                Gi_drp = ds.eval(
                    types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri+dr]]))
                Gi_drm = ds.eval(
                    types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri-dr]]))
                dGn = [(Gi_drp[i] - Gi_drm[i])/(2*dr) for i in [0, 1]]
                # Assert Symmetry
                np.testing.assert_array_equal(dGn[0], dGn[1])
                # Assert Values
                np.testing.assert_allclose(
                    dGa[0][:, 1, -1], dGn[0], rtol=1E-7, atol=1E-7)

    def test_dimer_polynomial(self):
        with DescriptorSet(['Au'], cutoff=7.) as ds:
            types = ['Au', 'Au']
            rss = [0.0, 0.0, 0.0]
            bohr2ang = 0.529177249
            etas = np.array([0.01, 0.1, 1.0])/bohr2ang**2

            ds.add_G2_functions(rss, etas, cuttype='polynomial')

            def cutfun(r):
                return (1
                        - 10.0 * (r/ds.cutoff)**3
                        + 15.0 * (r/ds.cutoff)**4
                        - 6.0 * (r/ds.cutoff)**5) * (r < ds.cutoff)

            dr = np.sqrt(np.finfo(float).eps)
            for ri in np.linspace(2, 8, 10):
                Gi = ds.eval(
                    types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri]]))
                # Assert Symmmetry
                np.testing.assert_array_equal(Gi[0], Gi[1])
                # Assert Values
                np.testing.assert_allclose(
                    Gi[0], np.exp(-etas*(ri-rss)**2)*cutfun(ri))
                # Derivatives
                dGa = ds.eval_derivatives(
                    types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri]]))
                # Assert Symmmetry
                np.testing.assert_array_equal(dGa[0], dGa[1])
                # Assert Values
                np.testing.assert_allclose(
                    dGa[0][:, 1, -1],
                    np.exp(-etas*(ri-rss)**2)*(
                        cutfun(ri)*2.0*(-etas)*(ri-rss)
                        + (- 30.0 * (ri**2/ds.cutoff**3)
                           + 60.0 * (ri**3/ds.cutoff**4)
                           - 30.0 * (ri**4/ds.cutoff**5)) *
                        (ri < ds.cutoff)))

                Gi_drp = ds.eval(
                    types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri+dr]]))
                Gi_drm = ds.eval(
                    types, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, ri-dr]]))
                dGn = [(Gi_drp[i] - Gi_drm[i])/(2*dr) for i in [0, 1]]
                # Assert Symmetry
                np.testing.assert_array_equal(dGn[0], dGn[1])
                # Assert Values
                np.testing.assert_allclose(
                    dGa[0][:, 1, -1], dGn[0], rtol=1E-7, atol=1E-7)

    def test_acetone(self):
        from scipy.optimize import approx_fprime
        # TODO: switch to custom implementation!
        x0 = np.array([0.00000,        0.00000,        0.00000,  # C
                       1.40704,        0.00902,       -0.67203,  # C
                       1.67062,       -0.92069,       -1.22124,  # H
                       2.20762,        0.06960,        0.11291,  # H
                       1.61784,        0.88539,       -1.32030,  # H
                      -1.40732,       -0.00378,       -0.67926,  # C
                      -1.65709,        0.91221,       -1.25741,  # H
                      -2.20522,       -0.03081,        0.10912,  # H
                      -1.64457,       -0.88332,       -1.31507,  # H
                       0.00000,       -0.00000,        1.20367])  # O
        types = ['C', 'C', 'H', 'H', 'H', 'C', 'H', 'H', 'H', 'O']

        with DescriptorSet(['C', 'H', 'O'], cutoff=7.0) as ds:
            radial_etas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
            rss = [0.0]*len(radial_etas)

            angular_etas = [0.0001, 0.003, 0.008]
            lambs = [1.0, -1.0]
            zetas = [1.0, 4.0]

            ds.add_G2_functions(rss, radial_etas)
            ds.add_G5_functions(angular_etas, zetas, lambs)
            f0 = ds.eval(types, x0.reshape((-1, 3)))
            df = ds.eval_derivatives(types, x0.reshape((-1, 3)))
            eps = np.sqrt(np.finfo(float).eps)

            for i in range(len(f0)):
                for j in range(len(f0[i])):
                    def f(x):
                        return np.array(
                            ds.eval(types, x.reshape((-1, 3))))[i][j]

                    np.testing.assert_array_almost_equal(
                        df[i][j],
                        approx_fprime(x0, f, epsilon=eps).reshape((-1, 3)))

    def test_eval_with_derivatives(self):
        xyzs = np.array([[1.19856,        0.00000,        0.71051],  # C
                         [2.39807,        0.00000,        0.00000],  # C
                         [2.35589,        0.00000,       -1.39475],  # C
                         [1.19865,        0.00000,       -2.09564],  # N
                         [0.04130,        0.00000,       -1.39453],  # C
                         [0.00000,        0.00000,        0.00000],  # C
                         [-0.95363,       0.00000,        0.52249],  # H
                         [3.35376,        0.00000,        0.51820],  # H
                         [3.26989,        0.00000,       -1.98534],  # H
                         [-0.87337,       0.00000,       -1.98400],  # H
                         [1.19077,        0.00000,        2.07481],  # O
                         [2.10344,        0.00000,        2.41504]])  # H
        types = ['C', 'C', 'C', 'N', 'C', 'C', 'H', 'H', 'H', 'H', 'O', 'H']

        with DescriptorSet(['C', 'N', 'H', 'O'], cutoff=7.0) as ds:
            # Parameters from Artrith and Kolpak Nano Lett. 2014, 14, 2670
            ds.add_Artrith_Kolpak_set()

            Gs_ref = ds.eval(types, xyzs)
            dGs_ref = ds.eval_derivatives(types, xyzs)
            Gs, dGs = ds.eval_with_derivatives(types, xyzs)
            np.testing.assert_allclose(Gs, Gs_ref)
            np.testing.assert_allclose(dGs, dGs_ref)

    def test_derivaties(self):
        with DescriptorSet(['Ni', 'Au'], cutoff=10.) as ds:
            pos = np.array([[0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0],
                            [0.0, -1.0, 0.0]])
            types = ['Ni', 'Ni', 'Au', 'Ni', 'Au']

            rss = [0.0, 0.0, 0.0]
            etas = [1.0, 0.01, 0.0001]

            ds.add_G2_functions(rss, etas)
            ds.add_G5_functions(etas, [1.0], [1.0])

            out_cpp = ds.eval(types, pos)
            analytical_derivatives = ds.eval_derivatives(types, pos)
            numerical_derivatives = np.zeros(
                (len(out_cpp), out_cpp[0].size, pos.size))
            dx = np.sqrt(np.finfo(float).eps)
            for i in range(pos.size):
                dpos = np.zeros(pos.shape)
                dpos[np.unravel_index(i, dpos.shape)] += dx
                numerical_derivatives[:, :, i] = (
                    np.array(ds.eval(types, pos+dpos))
                    - np.array(ds.eval(types, pos-dpos)))/(2*dx)

        np.testing.assert_array_almost_equal(
            numerical_derivatives.flatten(),
            np.array(analytical_derivatives).flatten())


if __name__ == '__main__':
    unittest.main()
