import unittest
from DescriptorLib.SymmetryFunctionSet import SymmetryFunctionSet as SymFunSet
import numpy as np
from itertools import product, combinations_with_replacement

class LibraryTest(unittest.TestCase):

    def test_cos_cutoff_functions(self):
        r_vec = np.linspace(0.1, 8, 101)
        geos = [[("F", np.array([0.0, 0.0, 0.0])),
                ("H", np.array([0.0, 0.0, ri]))] for ri in r_vec]
        with SymFunSet(["H", "F"], cutoff = 6.5) as sfs:
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
            np.testing.assert_allclose(np.array(dGs)[:,0,0,1,-1],
                0.5*(-np.sin(r_vec*np.pi/sfs.cutoff)*np.pi/sfs.cutoff)*(
                r_vec < sfs.cutoff))

if __name__ == '__main__':
    unittest.main()
