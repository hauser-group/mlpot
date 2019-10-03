from mlpot.calculators.mlcalculator import MLCalculator
from mlpot.calculators.gprcalculator import GPRCalculator
import numpy as np


class MaskedGPRCalculator(GPRCalculator):
    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, mask=None, **kwargs):
        GPRCalculator.__init__(self, restart, ignore_bad_restart_file, label,
                               atoms, **kwargs)
        self.mask = mask

    def add_data(self, atoms):
        # If the trainings set is empty: setup the numpy arrays
        if not self.atoms_train:
            self.n_atoms = len(atoms)
            self.n_dim = 3*sum(self.mask)
            self.x_train = np.zeros((0, self.n_dim))
            self.E_train = np.zeros(0)
            self.F_train = np.zeros(0)
        # else: check if the new atoms object has the same length as previous
        else:
            if not len(atoms) == self.n_atoms:
                raise ValueError('New data does not have the same number of '
                                 'atoms as previously added data.')

        # Call the super class routine after checking for empty trainings set!
        MLCalculator.add_data(self, atoms)
        self.x_train = np.append(
            self.x_train, self._transform_input(atoms), axis=0)
        # Call forces first in case forces and energy are calculated at the
        # same time by the calculator
        if self.mean_model is None:
            F = atoms.get_forces()[self.mask, :].flatten()
            E = atoms.get_potential_energy()
        else:
            F = ((atoms.get_forces()
                  - self.mean_model.get_forces(atoms=atoms))[self.mask, :]).flatten()
            E = (atoms.get_potential_energy()
                 - self.mean_model.get_potential_energy(atoms=atoms))
        self.E_train = np.append(self.E_train, E)
        self.F_train = np.append(self.F_train, F)

    def _transform_input(self, atoms):
        return atoms.get_positions()[self.mask, :].reshape(1, -1)

    def predict(self, atoms):
        # Prediction
        X_star = self._normalize_input(self._transform_input(atoms))
        y = self.alpha.dot(self.build_kernel_matrix(X_star=X_star))
        E = y[0] + self.intercept
        F = np.zeros((self.n_atoms, 3))
        F[self.mask, :] = -y[1:].reshape((-1, 3))
        if self.mean_model is not None:
            E += self.mean_model.get_potential_energy(atoms=atoms)
            F += self.mean_model.get_forces(atoms=atoms)
        return E, F

    def predict_var(self, atoms):
        X_star = self._normalize_input(self._transform_input(atoms))
        K_star = self.build_kernel_matrix(X_star=X_star)

        v = solve_triangular(self.L, K_star, lower=True)
        y_var = self.build_kernel_diagonal(X_star)
        y_var -= np.einsum('ij,ij->j', v, v)

        E_var = y_var[0]
        F_var = np.zeros((self.n_atoms, 3))
        F_var[self.mask, :] = y_var[1:].reshape((-1, 3))
        return E_var, F_var
