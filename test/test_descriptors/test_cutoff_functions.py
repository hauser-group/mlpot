import unittest
from mlpot.descriptors import DescriptorSet
import numpy as np
from itertools import product


class CutoffTest(unittest.TestCase):
    class CutoffTest(unittest.TestCase):

        def test_cutoff_function(self):
            r_vec = np.linspace(0.1, 8, 80)
            geos = [[('F', np.array([0.0, 0.0, 0.0])),
                    ('H', np.array([0.0, 0.0, ri]))] for ri in r_vec]
            with DescriptorSet(['H', 'F'], cutoff=6.5) as ds:
                for (t1, t2) in product(ds.atomtypes, repeat=2):
                    ds.add_two_body_descriptor(
                        t1, t2, 'BehlerG1', [], cuttype=self.cuttype)

                Gs = []
                dGs = []
                for geo in geos:
                    Gs.append(ds.eval_geometry(geo))
                    dGs.append(ds.eval_geometry_derivatives(geo))
                np.testing.assert_allclose(
                    np.array(Gs)[:, 0, 0],
                    self.function(r_vec, ds.cutoff), equal_nan=False)
                np.testing.assert_allclose(
                    np.array(dGs)[:, 0, 0, 1, -1],
                    self.function_derivative(r_vec, ds.cutoff),
                    atol=1e-12, equal_nan=False)


class ConstCutoffTest(CutoffTest.CutoffTest):
    cuttype = 'const'

    def function(self, r_vec, cutoff):
        return (1.0)*(r_vec < cutoff)

    def function_derivative(self, r_vec, cutoff):
        return np.zeros_like(r_vec)


class CosCutoffTest(CutoffTest.CutoffTest):
    cuttype = 'cos'

    def function(self, r_vec, cutoff):
        return 0.5*(1.0 + np.cos(r_vec*np.pi/cutoff))*(r_vec < cutoff)

    def function_derivative(self, r_vec, cutoff):
        return 0.5*(-np.sin(r_vec*np.pi/cutoff)*np.pi/cutoff)*(r_vec < cutoff)


class PolyCutoffTest(CutoffTest.CutoffTest):
    cuttype = 'polynomial'

    def function(self, r_vec, cutoff):
        return (1 - 10.0 * (r_vec/cutoff)**3
                + 15.0 * (r_vec/cutoff)**4
                - 6.0 * (r_vec/cutoff)**5)*(r_vec < cutoff)

    def function_derivative(self, r_vec, cutoff):
        return (- 30.0 * (r_vec/cutoff)**2/cutoff
                + 60.0 * (r_vec/cutoff)**3/cutoff
                - 30.0 * (r_vec/cutoff)**4/cutoff) * (r_vec < cutoff)


class TanhCutoffTest(CutoffTest.CutoffTest):
    cuttype = 'tanh'

    def function(self, r_vec, cutoff):
        return (np.tanh(1.0 - r_vec/cutoff)**3)*(r_vec < cutoff)

    def function_derivative(self, r_vec, cutoff):
        return ((-3*np.sinh(1.0 - r_vec/cutoff)**2) /
                (cutoff*np.cosh(1.0 - r_vec/cutoff)**4)) * (r_vec < cutoff)


class SmoothCutoffTest(CutoffTest.CutoffTest):
    cuttype = 'smooth'

    def function(self, r_vec, cutoff):
        output = np.zeros_like(r_vec)
        mask = r_vec < cutoff
        output[mask] = (1.0-np.exp(-cutoff/r_vec[mask])
                        / (np.exp(-cutoff/r_vec[mask])
                        + np.exp(-cutoff/(cutoff-r_vec[mask])))
                        )
        return output

    def function_derivative(self, r_vec, cutoff):
        output = np.zeros_like(r_vec)
        mask = r_vec < cutoff
        output[mask] = (cutoff*np.exp(
                        cutoff/(cutoff-r_vec[mask])
                        + cutoff/r_vec[mask])
                        * (cutoff**2 - 2*cutoff*r_vec[mask] + 2*r_vec[mask]**2)
                        / (r_vec[mask]**2*(np.exp(cutoff/(cutoff-r_vec[mask]))
                           + np.exp(cutoff/r_vec[mask]))**2
                           * (cutoff-r_vec[mask])**2))
        return output


class Smooth2CutoffTest(CutoffTest.CutoffTest):
    cuttype = 'smooth2'

    def function(self, r_vec, cutoff):
        output = np.zeros_like(r_vec)
        mask = r_vec < cutoff
        output[mask] = (np.exp(1.0 - 1.0/(1.0 - (r_vec[mask]/cutoff)**2)))
        return output

    def function_derivative(self, r_vec, cutoff):
        output = np.zeros_like(r_vec)
        mask = r_vec < cutoff
        output[mask] = ((-2.0*cutoff**2*r_vec[mask]
                        * np.exp(r_vec[mask]**2/(r_vec[mask]**2-cutoff**2))) /
                        (cutoff**2-r_vec[mask]**2)**2)
        return output


if __name__ == '__main__':
    unittest.main()
