from mlpot.calculators.mlcalculator import MLCalculator
from mlpot.nnpotentials import BPpotential
from mlpot.nnpotentials.utils import calculate_bp_indices
import numpy as np
import tensorflow as tf


class NNCalculator(MLCalculator):

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, C1=1.0, C2=1.0, lamb=1.0,
                 descriptor_set=None, layers=None, offsets=None,
                 normalize_input=False, model_dir=None, config=None,
                 opt_restarts=1, reset_fit=False, opt_method='L-BFGS-B',
                 maxiter=10000, maxcor=200, miniter=None,
                 e_tol=1e-3, f_tol=5e-2, **kwargs):
        MLCalculator.__init__(self, restart, ignore_bad_restart_file, label,
                              atoms, C1, C2, **kwargs)

        if descriptor_set is None:
            raise NotImplementedError('For now a descriptor set has to be ' +
                                      'passed to the constructor as no ' +
                                      'default set is implemented')
        else:
            self.descriptor_set = descriptor_set

        self.atomtypes = self.descriptor_set.atomtypes

        layers = layers or [[5, 5] for _ in self.atomtypes]
        offsets = offsets or [0.0 for _ in self.atomtypes]

        self.opt_restarts = opt_restarts
        self.reset_fit = reset_fit
        self.maxiter = maxiter
        self.miniter = miniter or maxcor
        self.e_tol = e_tol
        self.f_tol = f_tol

        self.model_dir = model_dir
        if normalize_input not in ['mean', 'min_max', 'norm', False]:
            raise NotImplementedError(
                'Unknown input normalization %s' % normalize_input)
        self.normalize_input = normalize_input
        # Depending on the type of input normalization these dicts hold
        # different values, for example the mean and the std for "mean" norm
        # or the minimum and the maximum-minimum difference for "min_max"
        self.Gs_norm1 = {}
        self.Gs_norm2 = {}
        for t in self.atomtypes:
            self.Gs_norm1[t] = np.zeros(self.descriptor_set.num_Gs[
                self.descriptor_set.type_dict[t]])
            self.Gs_norm2[t] = np.ones(self.descriptor_set.num_Gs[
                self.descriptor_set.type_dict[t]])

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.pot = BPpotential(
                self.atomtypes,
                [self.descriptor_set.num_Gs[self.descriptor_set.type_dict[t]]
                    for t in self.atomtypes], layers=layers,
                build_forces=True, offsets=offsets, precision=tf.float64)

            with self.graph.name_scope('train'):
                regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)
                reg_variables = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                # reg_variables = self.pot.variables
                reg_term = tf.contrib.layers.apply_regularization(
                    regularizer, reg_variables)
                self.sum_squared_error = tf.reduce_sum(
                    (self.pot.target-self.pot.E_predict)**2)
                self.sum_squared_error_force = tf.reduce_sum(
                    (self.pot.target_forces-self.pot.F_predict)**2)
                self.loss = tf.add(
                    .5*self.pot.mse,
                    .5*self.pot.mse_forces*(self.C2/self.C1)
                    + reg_term/self.C1, name='Loss')
                self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                    self.loss, method=opt_method,
                    options={'maxiter': maxiter, 'disp': False, 'ftol': 1e-14,
                             'gtol': 1e-14, 'maxcor': maxcor},
                    var_list=self.pot.variables)
                # [v for v in self.pot.variables if not 'b:' in v.name])
        self.session = tf.Session(config=config, graph=self.graph)
        self.session.run(tf.initializers.variables(self.pot.variables))

    def add_data(self, atoms):
        # If the trainings set is empty: setup the numpy arrays
        if not self.atoms_train:
            self.N_atoms = len(atoms)
            self.n_dim = 3*self.N_atoms
            self.E_train = np.zeros(0)
            self.F_train = []
            self.Gs = []
            self.dGs = []
            self.int_types = []
            self.indices = {t: np.zeros((0, 1)) for t in self.atomtypes}
            self.Gs_dict = {t: [] for t in self.atomtypes}
            self.dGs_dict = {t: [] for t in self.atomtypes}

        # Call the super class routine after checking for empty trainings set!
        MLCalculator.add_data(self, atoms)
        self.E_train = np.append(self.E_train, atoms.get_potential_energy())
        self.F_train.append(atoms.get_forces())

        self.int_types.append([self.descriptor_set.type_dict[ti] for ti in
                               atoms.get_chemical_symbols()])
        Gi, dGi = self.descriptor_set.eval_ase(atoms, derivatives=True)
        self.Gs.append(Gi)
        self.dGs.append(dGi)

        Gs_by_type, dGs_by_type = self._transform_input(atoms)
        for t in self.atomtypes:
            self.Gs_dict[t].extend(np.array(Gs_by_type[t]))
            self.dGs_dict[t].extend(np.array(dGs_by_type[t]))
            self.indices[t] = np.append(
                self.indices[t],
                (self.E_train.shape[0]-1) * np.ones((len(Gs_by_type[t]), 1)),
                axis=0)

    def _transform_input(self, atoms):
        types = atoms.get_chemical_symbols()
        Gs, dGs = self.descriptor_set.eval_ase(atoms, derivatives=True)
        Gs_by_type = {t: [] for t in self.atomtypes}
        dGs_by_type = {t: [] for t in self.atomtypes}
        for Gi, dGi, ti in zip(Gs, dGs, types):
            Gs_by_type[ti].append(Gi)
            dGs_by_type[ti].append(dGi)
        return Gs_by_type, dGs_by_type

    def fit(self):
        print('Fit called with %d geometries.' % len(self.atoms_train))
        # TODO: This could be streamlined even more by not using the
        # calculate_bp_indices function on the whole set but simply appending
        # to ann_inputs, indices and ann_derivs

        ann_inputs, indices, ann_derivs = calculate_bp_indices(
            len(self.atomtypes), self.Gs, self.int_types, dGs=self.dGs)

        for i, t in enumerate(self.atomtypes):
            np.testing.assert_equal(indices[i], self.indices[t])
            np.testing.assert_allclose(ann_inputs[i], self.Gs_dict[t])
            np.testing.assert_allclose(ann_derivs[i], self.dGs_dict[t])

        if self.normalize_input == 'mean':
            for i, t in enumerate(self.atomtypes):
                self.Gs_norm1[t] = np.mean(ann_inputs[i], axis=0)
                # Small offset for numerical stability
                self.Gs_norm2[t] = np.std(ann_inputs[i], axis=0) + 1E-6
        elif self.normalize_input == 'min_max':
            for i, t in enumerate(self.atomtypes):
                self.Gs_norm1[t] = np.min(ann_inputs[i], axis=0)
                # Small offset for numerical stability
                self.Gs_norm2[t] = (np.max(ann_inputs[i], axis=0) -
                                    np.min(ann_inputs[i], axis=0) + 1E-6)

        self.train_dict = {
            self.pot.target: self.E_train,
            self.pot.target_forces: self.F_train,
            self.pot.error_weights: np.ones(len(self.atoms_train))}
        for t in self.atomtypes:
            self.train_dict[self.pot.atom_indices[t]] = self.indices[t]
            if self.normalize_input == 'norm':
                norm_i = np.linalg.norm(self.Gs_dict[t], axis=1)
                self.train_dict[self.pot.atomic_contributions[t].input] = (
                    self.Gs_dict[t])/norm_i[:, np.newaxis]
                self.train_dict[
                    self.pot.atomic_contributions[t].derivatives_input
                    ] = np.einsum('ijkl,i->ijkl', self.dGs_dict[t], 1.0/norm_i)
            else:
                self.train_dict[self.pot.atomic_contributions[t].input] = (
                    self.Gs_dict[t]-self.Gs_norm1[t])/self.Gs_norm2[t]
                self.train_dict[
                    self.pot.atomic_contributions[t].derivatives_input
                    ] = np.einsum('ijkl,j->ijkl', self.dGs_dict[t],
                                  1.0/self.Gs_norm2[t])

        # Start with large minimum loss value
        min_loss_value = 1E20
        for i in range(self.opt_restarts):
            # Reset weights to random initialization:
            if (i > 0 or self.reset_fit or
                    (self.opt_restarts == 1 and self.reset_fit)):
                self.session.run(tf.initializers.variables(self.pot.variables))

            # Use tensorflow optimizer interface to create a function that
            # returns both RMSEs given a vector of weights
            eval_rmse = self.optimizer._make_eval_func(
                [self.pot.rmse, self.pot.rmse_forces], self.session,
                self.train_dict, [])

            # Dummy Exception which is raised to exit a running scipy optimizer
            class ConvergedNotAnError(Exception):
                pass

            self.opt_step = 0

            # Callback function that takes a vector of weights and checks if
            # the optimization is converged. Raises ConvergedNotAnError to stop
            # the scipy optimization if convergence criteria are met.
            def step_callback(packed_vars):
                if self.opt_step > self.miniter:
                    rmse, rmse_forces = eval_rmse(packed_vars)
                    # Check for convergence
                    if (rmse/self.N_atoms < self.e_tol
                            and rmse_forces < self.f_tol):
                        # Copied straight from tensorflow optimizer interface.
                        # Typically the variables in the graph are updated at
                        # the end of the optimization. Since we are interupting
                        # the optimization (by raising an Exception) the
                        # variables have to be updated at this point.
                        var_vals = [
                            packed_vars[packing_slice] for packing_slice
                            in self.optimizer._packing_slices]
                        self.session.run(
                            self.optimizer._var_updates,
                            feed_dict=dict(zip(
                                self.optimizer._update_placeholders, var_vals))
                        )
                        raise ConvergedNotAnError()
                self.opt_step = self.opt_step + 1

            # Optimize weights using scipy.minimize
            try:
                self.optimizer.minimize(self.session, self.train_dict,
                                        step_callback=step_callback)
            except ConvergedNotAnError:
                # This is actually the sucessful convergence
                print('Converged after %d steps.' % self.opt_step)

            loss_value, e_rmse, f_rmse = self.session.run(
                [self.loss, self.pot.rmse, self.pot.rmse_forces],
                self.train_dict)
            print('Finished optimization %d/%d after %d steps. ' % (
                i+1, self.opt_restarts, self.opt_step) +
                'Total loss = %f, RMSE energy = %f, RMSE forces = %f.' % (
                loss_value, e_rmse, f_rmse))
            # Save model parameters should this be a new minimum
            if loss_value < min_loss_value:
                # save loss value and parameters to restore minimum later
                min_loss_value = loss_value
                self.pot.saver.save(self.session,
                                    self.model_dir+'min_model.ckpt')

            if e_rmse/self.N_atoms < self.e_tol and f_rmse < self.f_tol:
                break

        self.pot.saver.restore(self.session, self.model_dir+'min_model.ckpt')
        e_rmse, f_rmse = self.session.run(
            [self.pot.rmse, self.pot.rmse_forces], self.train_dict)
        print('Fit finished. Final RMSE energy = '
              '%f, RMSE force = %f.' % (e_rmse, f_rmse))

    def predict(self, atoms):
        int_types = [self.descriptor_set.type_dict[ti] for ti in
                     atoms.get_chemical_symbols()]
        Gs, dGs = self.descriptor_set.eval_ase(atoms, derivatives=True)
        ann_inputs, indices, ann_derivs = calculate_bp_indices(
            len(self.atomtypes), [Gs], [int_types], dGs=[dGs])
        Gs, dGs = self._transform_input(atoms)
        for i, t in enumerate(self.atomtypes):
            np.testing.assert_equal(indices[i],
                                    np.zeros((len(Gs[t]), 1)))
            np.testing.assert_allclose(ann_inputs[i], Gs[t])
            np.testing.assert_allclose(ann_derivs[i], dGs[t])

        eval_dict = {self.pot.target: np.zeros(1),
                     self.pot.target_forces: np.zeros((1, len(atoms), 3))}
        for t in self.atomtypes:
            eval_dict[self.pot.atom_indices[t]] = np.zeros((len(Gs[t]), 1))
            if self.normalize_input == 'norm':
                norm_i = np.linalg.norm(Gs[t], axis=1)
                eval_dict[self.pot.atomic_contributions[t].input] = (
                    Gs[t])/norm_i[:, np.newaxis]
                eval_dict[
                    self.pot.atomic_contributions[t].derivatives_input
                    ] = np.einsum('ijkl,i->ijkl', dGs[t], 1.0/norm_i)
            else:
                eval_dict[self.pot.atomic_contributions[t].input] = (
                    Gs[t]-self.Gs_norm1[t])/self.Gs_norm2[t]
                eval_dict[
                    self.pot.atomic_contributions[t].derivatives_input
                    ] = np.einsum('ijkl,j->ijkl', dGs[t],
                                  1.0/self.Gs_norm2[t])

        E = self.session.run(self.pot.E_predict, eval_dict)[0]
        F = self.session.run(self.pot.F_predict, eval_dict)[0]
        return E, F

    def get_params(self):
        return {'atoms_train': self.atoms_train,
                'normalize_input': self.normalize_input,
                'Gs': self.Gs,
                'dGs': self.dGs,
                'Gs_norm': self.Gs_norm,
                'dGs_norm': self.dGs_norm,
                'Gs_norm1': self.Gs_norm1,
                'Gs_norm2': self.Gs_norm2,
                'model_dir': self.pot.saver.save(
                    self.session, self.model_dir+'model.ckpt')}

    def set_params(self, **params):
        self.atoms_train = params['atoms_train']
        self.normalize_input = params['normalize_input']
        self.Gs = params['Gs']
        self.dGs = params['dGs']
        self.Gs_norm = params['Gs_norm']
        self.dGs_norm = params['dGs_norm']
        self.Gs_norm1 = params['Gs_norm1']
        self.Gs_norm2 = params['Gs_norm2']
        self.n_dim = 3*len(self.atoms_train[0])
        self.pot.saver.restore(self.session, params['model_dir'])

    def close(self):
        self.session.close()
