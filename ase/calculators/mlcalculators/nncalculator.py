from ase.calculators.mlcalculators.mlcalculator import MLCalculator
from NNpotentials import BPpotential
from NNpotentials.utils import calculate_bp_indices
import numpy as np
import tensorflow as tf

class NNCalculator(MLCalculator):

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, C1=1.0, C2=1.0, lamb=1.0,
                 descriptor_set=None, layers=None, offsets=None,
                 normalize_input=False, model_dir=None, opt_restarts=1,
                 reset_fit=True, maxiter=1000, opt_method='L-BFGS-B', **kwargs):
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

        self.opt_restarts = opt_restarts
        self.reset_fit = reset_fit

        self.model_dir = model_dir
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
                regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)
                #reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_variables = self.pot.variables
                reg_term = tf.contrib.layers.apply_regularization(
                    regularizer, reg_variables)
                self.sum_squared_error = tf.reduce_sum(
                    (self.pot.target-self.pot.E_predict)**2)
                self.sum_squared_error_force = tf.reduce_sum(
                    (self.pot.target_forces-self.pot.F_predict)**2)
                self.loss = tf.add(.5*self.sum_squared_error,
                    .5*self.sum_squared_error_force*(self.C2/self.C1)
                    + reg_term/self.C1,
                    name='Loss')
                self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                    self.loss, method=opt_method, options={'maxiter': maxiter,
                    'disp': False, 'ftol':1E-20, 'gtol':1E-10},
                    var_list = self.pot.variables)#[v for v in self.pot.variables if not 'b:' in v.name])
        self.session = tf.Session(graph=self.graph)
        self.session.run(tf.initializers.variables(self.pot.variables))

    def fit(self, atoms_list):
        print('Fit called with %d geometries.'%len(atoms_list))
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

        ann_inputs, indices, ann_derivs = calculate_bp_indices(
            len(self.atomtypes), Gs, int_types, dGs=dGs)

        if self.normalize_input:
            for i, t in enumerate(self.atomtypes):
                self.Gs_mean[t] = np.mean(ann_inputs[i], axis=0)
                # Small offset for numerical stability
                self.Gs_std[t] = np.std(ann_inputs[i], axis=0) + 1E-6

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

        # Start with large minimum loss value
        min_loss_value = 1E20
        for i in range(self.opt_restarts):
            # Reset weights to random initialization:
            if (i > 0 or self.reset_fit or
                  (self.opt_restarts == 1 and self.reset_fit)):
                self.session.run(tf.initializers.variables(self.pot.variables))
            # Optimize weights using scipy.minimize
            self.optimizer.minimize(self.session, train_dict)

            loss_value, e_rmse, f_rmse = self.session.run(
                [self.loss, self.pot.rmse, self.pot.rmse_forces], train_dict)
            print('Finished optimization %d/%d. '%(i+1,self.opt_restarts) +
                'Total loss = %f, RMSE energy = %f, RMSE forces = %f.'%(
                    loss_value, e_rmse, f_rmse))
            if loss_value < min_loss_value:
                # save loss value and parameters to restore minimum later
                min_loss_value = loss_value
                self.pot.saver.save(self.session,
                    self.model_dir+'min_model.ckpt')

        self.pot.saver.restore(self.session, self.model_dir+'min_model.ckpt')
        e_rmse, f_rmse = self.session.run(
            [self.pot.rmse, self.pot.rmse_forces], train_dict)
        print('Fit finished. Final RMSE energy = '
            '%f, RMSE force = %f.'%(e_rmse, f_rmse))

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

    def get_params(self):
        params = {'normalize_input':self.normalize_input,
            'Gs_mean':self.Gs_mean, 'Gs_std':self.Gs_std,
            'model_dir':self.pot.saver.save(
                self.session, self.model_dir+'model.ckpt')}
        return params

    def set_params(self, **params):
        self.normalize_input = params['normalize_input']
        self.Gs_mean = params['Gs_mean']
        self.Gs_std = params['Gs_std']
        self.pot.saver.restore(self.session, params['model_dir'])

    def close(self):
        self.session.close()
