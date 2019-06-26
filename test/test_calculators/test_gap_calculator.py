import unittest
import numpy as np

from ase.atoms import Atoms
from ase.calculators.emt import EMT
from itertools import combinations_with_replacement
from mlpot.calculators.gapcalculator import GAPCalculator
from mlpot.descriptors.SymmetryFunctionSet import SymmetryFunctionSet
from mlpot.kernels import RBFKernel


class GapCalculatorTest(unittest.TestCase):
    def test_co_potential_curve(self):
        direction = np.array([1., 2., 3.])
        direction /= np.linalg.norm(direction)
        r_train = np.linspace(0.5, 7, 11)
        energies_train = []
        images_train = []
        for ri in r_train:
            image = Atoms(
                ['C', 'O'],
                positions=np.array([-0.5*ri*direction, 0.5*ri*direction]))
            image.set_calculator(EMT())
            energies_train.append(image.get_potential_energy())
            images_train.append(image)
        cut = 6.5

        with SymmetryFunctionSet(["C", "O"], cutoff=cut) as sfs:
            # Parameters from Artrith and Kolpak Nano Lett. 2014, 14, 2670
            etas = [0.0009, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2]
            rss = np.linspace(0, 6.5, 7)
            for t1 in sfs.atomtypes:
                for t2 in sfs.atomtypes:
                    for eta, rs in zip(etas, rss):
                        sfs.add_TwoBodySymmetryFunction(
                            t1, t2, 'BehlerG2', [eta, rs], cuttype='cos')

            ang_etas = [0.0001, 0.003, 0.008]
            zetas = [1.0, 4.0]
            for ti in sfs.atomtypes:
                for (tj, tk) in combinations_with_replacement(
                        sfs.atomtypes, 2):
                    for eta in ang_etas:
                        for lamb in [-1.0, 1.0]:
                            for zeta in zetas:
                                sfs.add_ThreeBodySymmetryFunction(
                                    ti, tj, tk, "BehlerG3",
                                    [lamb, zeta, eta], cuttype='cos')

            kernel = RBFKernel(constant=100.0, length_scale=1e-2)
            calc = GAPCalculator(descriptor_set=sfs, kernel=kernel,
                                 C1=1e8, C2=1e8, opt_restarts=0)

            [calc.add_data(image) for image in images_train]
            calc.fit()
            np.testing.assert_allclose(
                energies_train,
                [calc.predict(image)[0] for image in images_train])


if __name__ == '__main__':
    unittest.main()
