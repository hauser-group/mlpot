import unittest
import numpy as np

from ase.atoms import Atoms
from ase.calculators.emt import EMT
from mlpot.calculators.gprcalculator import GPRCalculator
from mlpot.kernels import RBFKernel


class GPRCalculatorTest(unittest.TestCase):
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

        kernel = RBFKernel(constant=100.0, length_scale=1e-1)
        calc = GPRCalculator(kernel=kernel, C1=1e8, C2=1e8, opt_restarts=0)

        [calc.add_data(im) for im in images_train]
        calc.fit()
        np.testing.assert_allclose(
            energies_train,
            [calc.predict(im)[0] for im in images_train])

    def test_prediction_variance(self):
        direction = np.array([1., 2., 3.])
        direction /= np.linalg.norm(direction)
        r_train = [0.7, 1.7]
        images_train = []
        for ri in r_train:
            image = Atoms(
                ['C', 'O'],
                positions=np.array([-0.5*ri*direction, 0.5*ri*direction]))
            image.set_calculator(EMT())
            images_train.append(image)

        r_test = np.linspace(0.5, 1.9, 101)
        images_test = []
        for ri in r_test:
            image = Atoms(
                ['C', 'O'],
                positions=np.array([-0.5*ri*direction, 0.5*ri*direction]))
            image.set_calculator(EMT())
            images_test.append(image)

        kernel = RBFKernel(constant=100., length_scale=.1)
        calc = GPRCalculator(kernel=kernel, C1=1E8, C2=1E8, opt_restarts=0)
        [calc.add_data(im) for im in images_train]

        calc.fit()
        prediction_var = [calc.predict_var(im)[0] for im in images_test]
        max_var = np.argmax(prediction_var)
        np.testing.assert_equal(r_test[max_var], 1.2)


if __name__ == '__main__':
    unittest.main()
