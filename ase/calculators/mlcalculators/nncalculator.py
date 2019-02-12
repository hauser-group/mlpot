from ase.calculators.mlcalculators.mlcalculator import MLCalculator
from NNpotentials import BPpotential
from NNpotentials.utils import calculate_bp_indices
import numpy as np
import tensorflow as tf

class NNCalculator(MLCalculator):

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, C1=1.0, C2=1.0,
                 descriptor_set=None, layers=None, offsets=None,
                 normalize_input=False, reset_fit=False, **kwargs):
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

        self.reset_fit = reset_fit
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
                loss = tf.add(self.C1*self.pot.rmse,
                    self.C2*self.pot.rmse_forces + reg_term, name='Loss')
                self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss,
                    method='L-BFGS-B')
        self.session = tf.Session(graph=self.graph)
        self.session.run(tf.initializers.variables(self.pot.variables))

    def fit(self, atoms_list):
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
            Gi, dGi = self.descriptor_set.eval_with_derivatives(int_types[-1],
                atoms.get_positions())
            Gs.append(Gi)
            dGs.append(dGi)

        print('Fit called with %d geometries. E_max = %f E_min=%f'%(
            len(atoms_list), np.max(energies), np.min(energies)))
        ann_inputs, indices, ann_derivs = calculate_bp_indices(
            len(self.atomtypes), Gs, int_types, dGs=dGs)

        #print(forces)
        #print(indices[0])
        #print(indices[1])
        if self.normalize_input:
            for i, t in enumerate(self.atomtypes):
                self.Gs_mean[t] = np.mean(ann_inputs[i], axis=0)
                # Small offset for numerical stability
                self.Gs_std[t] = np.std(ann_inputs[i], axis=0) + 1E-6

        #print(self.Gs_mean['C'])
        #print(self.Gs_mean['H'])
        train_dict = {self.pot.target: energies,
            self.pot.target_forces: forces,
            self.pot.error_weights: np.ones(len(atoms_list))}
        for i, t in enumerate(self.atomtypes):
            train_dict[self.pot.atomic_contributions[t].input] = (
                ann_inputs[i]-self.Gs_mean[t])/self.Gs_std[t]
            train_dict[self.pot.atom_indices[t]] = indices[i]
            train_dict[
                self.pot.atomic_contributions[t].derivatives_input
                ] = np.einsum('ijkl,j->ijkl', ann_derivs[i], 1.0/self.Gs_std[t])
        #print(train_dict[self.pot.atomic_contributions['C'].input])
        #print(train_dict[self.pot.atomic_contributions['H'].input])
        if self.reset_fit:
            self.session.run(tf.initializers.variables(self.pot.variables))
        self.optimizer.minimize(self.session, train_dict)
        e_rmse, f_rmse = self.session.run(
            [self.pot.rmse, self.pot.rmse_forces], train_dict)
        print('fit finished with energy rmse '
            '%f and gradient rmse %f'%(e_rmse, f_rmse))

        self.fitted = True

    def predict(self, atoms):
        int_types = [self.descriptor_set.type_dict[ti] for ti in
            atoms.get_chemical_symbols()]
        Gs, dGs = self.descriptor_set.eval_with_derivatives(int_types,
            atoms.get_positions())
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

        E = self.session.run(self.pot.E_predict, eval_dict)[0]
        F = self.session.run(self.pot.F_predict, eval_dict)[0]
        return E, F

    def get_params(self, model_dir):
        params = {'normalize_input':self.normalize_input,
            'Gs_mean':self.Gs_mean, 'Gs_std':self.Gs_std,
            'model_dir':self.pot.saver.save(
                self.session, model_dir+'model.ckpt')}
        return params

    def set_params(self, **params):
        self.normalize_input = params['normalize_input']
        self.Gs_mean = params['Gs_mean']
        self.Gs_std = params['Gs_std']
        self.pot.saver.restore(self.session, params['model_dir'])

    def close(self):
        self.descriptor_set.close()
        self.session.close()
