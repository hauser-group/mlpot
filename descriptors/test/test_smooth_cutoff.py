import unittest
from DescriptorLib.SymmetryFunctionSet import SymmetryFunctionSet as SymFunSet
import numpy as np
from itertools import product, combinations_with_replacement

class LibraryTest(unittest.TestCase):
    def test_smooth_cutoff_functions(self):
        r_vec = np.linspace(0.1, 7, 101)
        geos = [[("F", np.array([0.0, 0.0, 0.0])),
                ("H", np.array([0.0, 0.0, ri]))] for ri in r_vec]
        cut = 6.5

        with SymFunSet(["H", "F"], cutoff = cut) as sfs:
            for (t1, t2) in product(sfs.atomtypes, repeat = 2):
                sfs.add_TwoBodySymmetryFunction(
                    t1, t2, "BehlerG0", [], cuttype = "smooth")

            Gs = []
            dGs = []
            for geo in geos:
                Gs.append(sfs.eval_geometry(geo))
                dGs.append(sfs.eval_geometry_derivatives(geo))
            np.testing.assert_allclose(np.array(Gs)[:,0,0],
                (1.0-np.exp(-cut/r_vec)/(np.exp(-cut/r_vec)+
                np.exp(-cut/(cut-r_vec))))*(r_vec < cut))
            np.testing.assert_allclose(np.array(dGs)[:,0,0,1,-1],
                (cut*np.exp(cut/(cut-r_vec)+cut/r_vec)*
                (cut**2-2.*cut*r_vec+2.*r_vec**2)/
                (r_vec**2*(np.exp(cut/(cut-r_vec))+np.exp(cut/r_vec))**2*
                (cut-r_vec)**2))*(r_vec < cut), atol = 1E-7)

if __name__ == '__main__':
    unittest.main()
