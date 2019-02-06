from ase.calculators.mlcalculators.mlcalculator import MLCalculator
from NNpotentials import BPpotential
from NNpotentials.utils import calculate_bp_indices
import numpy as np
import tensorflow as tf

class NNCalculator(MLCalculator):

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, C1=1.0, C2=1.0,
                 descriptor_set=None, layers=None, offsets=None,
                 normalize_input=True, **kwargs):
        MLCalculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms, C1, C2, **kwargs)

        if descriptor_set == None:
            raise NotImplementedError('For now a descriptor set has to be' +
                ' passed to the constructor as no default set is implemented')
        else:
            self.descriptor_set = descriptor_set

        self.atomtypes = self.descriptor_set.atomtypes

        if layers == None:
            layers = [[5,5] for _ in self.atomtypes]
        if offsets == None:
            offsets = [0.0 for _ in self.atomtypes]

        self.normalize_input = normalize_input
        self.Gs_mean = {}
        self.Gs_std = {}
        for t in self.atomtypes:
            self.Gs_mean[t] = np.zeros(self.descriptor_set.num_Gs[
                self.descriptor_set.type_dict[t]])
            self.Gs_std[t] = np.ones(self.descriptor_set.num_Gs[
                self.descriptor_set.type_dict[t]])

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.pot = BPpotential(self.atomtypes,
                [self.descriptor_set.num_Gs[self.descriptor_set.type_dict[t]]
                    for t in self.atomtypes], layers = layers,
                build_forces = True, offsets = offsets, precision = tf.float64)

            with self.graph.name_scope('train'):
                regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_term = tf.contrib.layers.apply_regularization(
                    regularizer, reg_variables)
                self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                    self.C1*self.pot.rmse + self.C2*self.pot.rmse_forces + reg_term,
                    method='L-BFGS-B')
        self.session = tf.Session(graph=self.graph)

    def fit(self, atoms_list, reset=True):
        Gs = []
        dGs = []
        energies = []
        forces = []
        int_types = []

        for atoms in atoms_list:
            energies.append(atoms.get_potential_energy())
            forces.append(atoms.get_forces())
            int_types.append([self.descriptor_set.type_dict[ti] for ti in
                atoms.get_chemical_symbols()])
            Gi, dGi = self.descriptor_set.eval_ase(atoms, derivatives=True)
            Gs.append(Gi)
            dGs.append(dGi)
        ann_inputs, indices, ann_derivs = calculate_bp_indices(
            len(self.atomtypes), Gs, int_types, dGs=dGs)
        if self.normalize_input:
            for i, t in enumerate(self.atomtypes):
                self.Gs_mean[t] = np.mean(ann_inputs[i], axis=0)
                # Small offset for numerical stability
                self.Gs_std[t] = np.std(ann_inputs[i], axis=0) + 1E-6
        train_dict = {self.pot.target: energies,
            self.pot.target_forces: forces,
            self.pot.error_weights: np.ones(len(Gs))}
        for i, t in enumerate(self.atomtypes):
            train_dict[self.pot.atomic_contributions[t].input] = (
                ann_inputs[i]-self.Gs_mean[t])/self.Gs_std[t]
            train_dict[self.pot.atom_indices[t]] = indices[i]
            print(ann_derivs[i].shape)
            train_dict[
                self.pot.atomic_contributions[t].derivatives_input
                ] = np.einsum('ijkl,j->ijkl', ann_derivs[i], 1.0/self.Gs_std[t])
        if self.reset:
            self.session.run(tf.initializers.variables(self.pot.variables))
        self.optimizer.minimize(self.session, train_dict)
        e_rmse, f_rmse = self.session.run(
            [self.pot.rmse, self.pot.rmse_forces], train_dict)
        print('fit finished with energy rmse '
            '%f and gradient rmse %f'%(e_rmse, f_rmse))

        self.fitted = True

    def predict(self, atoms):
        Gs, dGs = self.descriptor_set.eval_ase(atoms, derivatives=True)
        int_types = [self.descriptor_set.type_dict[ti] for ti in
            atoms.get_chemical_symbols()]
        ann_inputs, indices, ann_derivs = calculate_bp_indices(
            len(self.atomtypes), [Gs], [int_types], dGs = [dGs])

        eval_dict = {self.pot.target: np.zeros(1),
            self.pot.target_forces: np.zeros((1, len(atoms), 3))}
        for i, t in enumerate(self.atomtypes):
            eval_dict[self.pot.atomic_contributions[t].input] = (
                ann_inputs[i]-self.Gs_mean[t])/self.Gs_std[t]
            eval_dict[self.pot.atom_indices[t]] = indices[i]
            eval_dict[
                self.pot.atomic_contributions[t].derivatives_input
                ] = np.einsum('ijkl,j->ijkl', ann_derivs[i], 1.0/self.Gs_std[t])

        E = self.session.run(self.pot.E_predict, eval_dict)
        F = self.session.run(self.pot.F_predict, eval_dict)
        return E, F

    def get_params(self):
        return self.session.run(self.pot.variables)

    def set_params(self, **params):
        pass

    def close(self):
        self.descriptor_set.close()
        self.session.close()
