from ase.optimize.optimize import Optimizer
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import QuasiNewton
import numpy as np


class MLOptimizer(Optimizer):

    def __init__(self, atoms, ml_calc, restart=None, logfile='-',
                 trajectory=None, master=None, force_consistent=None,
                 optimizer=QuasiNewton, maxstep=0.2, check_downhill=True,
                 callback_before_fit=None, callback_after_ml_opt=None):
        self.ml_calc = ml_calc
        self.optimizer = optimizer
        self.check_downhill = check_downhill
        self.previous_energy = None
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
        current_position = self.atoms.get_positions()

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
        # Clear previous results since fit parameters have changed
        self.ml_calc.results = {}
        # If last step was downhill start from the current_positions (this is
        # necessary as the last step might have been scaled down),
        # otherwise start from self.previous_position
        if self.check_downhill:
            current_energy = self.atoms.get_potential_energy()
            if (self.previous_energy is not None and
                    current_energy > self.previous_energy):
                print('Last step was uphill! Resetting position.')
                self.ml_atoms.set_positions(self.previous_position)
            else:  # Save current energy and position for next downhill check
                self.ml_atoms.set_positions(current_position)
                self.previous_energy = current_energy
                self.previous_position = self.atoms.get_positions()
        else:
            self.ml_atoms.set_positions(current_position)

        opt = self.optimizer(self.ml_atoms, logfile=None)
        opt.run(self.fmax)
        if self.callback_after_ml_opt is not None:
            self.callback_after_ml_opt(self.ml_calc)

        new_position = self.ml_atoms.get_positions()
        step = new_position - current_position
        step_length = np.linalg.norm(step)
        if step_length > self.maxstep:
            step *= self.maxstep / step_length
        self.atoms.set_positions(current_position + step)

        self.iteration += 1
        self.dump((self.iteration))
