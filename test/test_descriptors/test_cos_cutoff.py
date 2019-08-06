import unittest
from mlpot.descriptors import DescriptorSet
import numpy as np
from itertools import product


class LibraryTest(unittest.TestCase):

    def test_cos_cutoff_functions(self):
        r_vec = np.linspace(0.1, 8, 101)
        geos = [[("F", np.array([0.0, 0.0, 0.0])),
                ("H", np.array([0.0, 0.0, ri]))] for ri in r_vec]
        with DescriptorSet(["H", "F"], cutoff=6.5) as ds:
            for (t1, t2) in product(ds.atomtypes, repeat=2):
                ds.add_two_body_descriptor(
                    t1, t2, "BehlerG0", [], cuttype="cos")

            Gs = []
            dGs = []
            for geo in geos:
                Gs.append(ds.eval_geometry(geo))
                dGs.append(ds.eval_geometry_derivatives(geo))
            np.testing.assert_allclose(
                np.array(Gs)[:, 0, 0],
                0.5*(1.0+np.cos(r_vec*np.pi/ds.cutoff))*(r_vec < ds.cutoff))
            np.testing.assert_allclose(
                np.array(dGs)[:, 0, 0, 1, -1],
                0.5*(-np.sin(r_vec*np.pi/ds.cutoff)*np.pi/ds.cutoff)*(
                    r_vec < ds.cutoff))


if __name__ == '__main__':
    unittest.main()
