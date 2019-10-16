from ase.optimize.optimize import Optimizer
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import QuasiNewton
import numpy as np


class MLOptimizer(Optimizer):

    def __init__(self, atoms, ml_calc, restart=None, logfile='-',
                 trajectory=None, master=None, force_consistent=None,
                 optimizer=QuasiNewton, maxstep=0.2, callback_before_fit=None,
                 callback_after_ml_opt=None):
        self.ml_calc = ml_calc
        self.optimizer = optimizer
        self.callback_before_fit = callback_before_fit
        self.callback_after_ml_opt = callback_after_ml_opt
        self.maxstep = maxstep
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master,
                           force_consistent=force_consistent)

    def initialize(self):
        self.iteration = 0
        self.ml_atoms = self.atoms.copy()
        self.ml_atoms.set_calculator(self.ml_calc)

    def step(self, f=None):
        previous_position = self.atoms.get_positions()

        if f is None:
            f = self.atoms.get_forces()

        atoms_train = self.atoms.copy()
        atoms_train.set_calculator(SinglePointCalculator(
            forces=self.atoms.get_forces(apply_constraint=False),
            energy=self.atoms.get_potential_energy(),
            atoms=atoms_train))
        self.ml_calc.add_data(atoms_train)

        if self.callback_before_fit is not None:
            self.callback_before_fit(self.ml_calc)

        self.ml_calc.fit()
        # Clear previous results
        self.ml_calc.results = {}
        self.ml_atoms.set_positions(previous_position)

        opt = self.optimizer(self.ml_atoms, logfile=None)
        opt.run(self.fmax)
        if self.callback_after_ml_opt is not None:
            self.callback_after_ml_opt(self.ml_calc)

        new_position = self.ml_atoms.get_positions()
        step = new_position - previous_position
        step_length = np.linalg.norm(step)
        if step_length > self.maxstep:
            step *= self.maxstep / step_length
        self.atoms.set_positions(previous_position + step)

        self.iteration += 1
        self.dump((self.iteration))
