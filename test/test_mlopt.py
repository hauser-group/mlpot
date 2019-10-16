import numpy as np
import unittest
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from mlpot.calculators.gprcalculator import GPRCalculator
from mlpot.kernels import RBFKernel, Rescaling
from mlpot.mlopt import MLOptimizer


class H2Test(unittest.TestCase):

    def test_h2(self):
        mol = molecule('H2')
        mol.set_calculator(EMT())
        kernel = Rescaling(RBFKernel())
        ml_calc = GPRCalculator(kernel=kernel, opt_restarts=1,
                                normalize_y='max+10')

        r_test = np.linspace(0.6, 2.0)
        atoms_test = [Atoms(['H', 'H'],
                            positions=np.array([[0, 0, 0.5*ri],
                                                [0, 0, -0.5*ri]])
                            ) for ri in r_test]
        E_test = [EMT().get_potential_energy(atoms) for atoms in atoms_test]
        # import matplotlib.pyplot as plt

        def callback_after(ml_calc):
            plt.plot(r_test, E_test)
            plt.plot(r_test,
                     [ml_calc.predict(atoms)[0] for atoms in atoms_test])
            for atoms in ml_calc.atoms_train:
                xyzs = atoms.get_positions()
                r = xyzs[0, 2] - xyzs[1, 2]
                print(r)
                plt.plot(r, ml_calc.predict(atoms)[0], 'o')
            plt.show()

        opt = MLOptimizer(mol, ml_calc)#, callback_after_ml_opt=callback_after)
        opt.run()

    def test_butadiene(self):
        mol = molecule('butadiene')
        mol.set_calculator(EMT())
        kernel = Rescaling(RBFKernel())
        ml_calc = GPRCalculator(kernel=kernel, opt_restarts=1,
                                normalize_y='max+10')

        opt = MLOptimizer(mol, ml_calc)
        opt.run()


if __name__ == '__main__':
    unittest.main()
