import unittest
import numpy as np

from ase.atoms import Atoms
from ase.calculators.emt import EMT
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
            sfs.add_Artrith_Kolpak_set()

            kernel = RBFKernel(constant=100.0, length_scale=1e-2)
            calc = GAPCalculator(descriptor_set=sfs, kernel=kernel,
                                 C1=1e8, C2=1e8, opt_restarts=0)

            [calc.add_data(im) for im in images_train]
            calc.fit()
            np.testing.assert_allclose(
                energies_train,
                [calc.predict(im)[0] for im in images_train])


if __name__ == '__main__':
    unittest.main()
