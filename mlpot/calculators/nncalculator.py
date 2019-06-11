from mlpot.calculators.mlcalculator import MLCalculator
from mlpot.nnpotentials import BPpotential
from mlpot.nnpotentials.utils import calculate_bp_indices
import numpy as np
import tensorflow as tf

class ConvergedNotAnError(Exception):
    pass

class NNCalculator(MLCalculator):

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, C1=1.0, C2=1.0, lamb=1.0,
                 descriptor_set=None, layers=None, offsets=None,
                 normalize_input=False, model_dir=None, config=None,
                 opt_restarts=1, reset_fit=True, opt_method='L-BFGS-B',
                 maxiter=5000, maxcor=200, e_tol=1e-3, f_tol=5e-2, adam_thresh=100,
                 adam_learning_rate=5e-3, adam_epsilon=1e-08,
                 adam_maxiter=10000, **kwargs):
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
        self.maxiter = maxiter
        self.e_tol = e_tol
        self.f_tol = f_tol
        self.adam_maxiter = adam_maxiter
        self.adam_thresh = adam_thresh

        self.model_dir = model_dir
        if not normalize_input in ['mean', 'min_max', 'norm', False]:
            raise NotImplementedError(
                'Unknown input normalization %s'%normalize_input)
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
            self.pot = BPpotential(self.atomtypes,
                [self.descriptor_set.num_Gs[self.descriptor_set.type_dict[t]]
                    for t in self.atomtypes], layers = layers,
                build_forces = True, offsets = offsets, precision = tf.float64)

            with self.graph.name_scope('train'):
                regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)
                reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                #reg_variables = self.pot.variables
                reg_term = tf.contrib.layers.apply_regularization(
                    regularizer, reg_variables)
                self.sum_squared_error = tf.reduce_sum(
                    (self.pot.target-self.pot.E_predict)**2)
                self.sum_squared_error_force = tf.reduce_sum(
                    (self.pot.target_forces-self.pot.F_predict)**2)
                self.loss = tf.add(.5*self.pot.mse,
                    .5*self.pot.mse_forces*(self.C2/self.C1)
                    + reg_term/self.C1, name='Loss')
                self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                    self.loss, method=opt_method,
                    options={'maxiter':maxiter, 'disp':False, 'gtol':1e-10,
                        'maxcor':maxcor},
                    var_list = self.pot.variables)#[v for v in self.pot.variables if not 'b:' in v.name])
                self.AdamOpt = tf.train.AdamOptimizer(
                    learning_rate=adam_learning_rate, epsilon=adam_epsilon)
                self.train = self.AdamOpt.minimize(self.loss)
                self.init_adam = tf.initializers.variables(self.AdamOpt.variables())
        self.session = tf.Session(config=config, graph=self.graph)
        self.session.run(tf.initializers.variables(self.pot.variables))
        self.session.run(self.init_adam)

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

        # Call the super class routine after checking for empty trainings set!
        MLCalculator.add_data(self, atoms)
        self.E_train = np.append(self.E_train, atoms.get_potential_energy())
        self.F_train.append(atoms.get_forces())

        self.int_types.append([self.descriptor_set.type_dict[ti] for ti in
            atoms.get_chemical_symbols()])
        Gi, dGi = self.descriptor_set.eval_ase(atoms, derivatives=True)
        self.Gs.append(Gi)
        self.dGs.append(dGi)

    def fit(self):
        global opt_step
        print('Fit called with %d geometries.'%len(self.atoms_train))
        # TODO: This could be streamlined even more by not using the
        # calculate_bp_indices function on the whole set but simply appending
        # to ann_inputs, indices and ann_derivs

        ann_inputs, indices, ann_derivs = calculate_bp_indices(
            len(self.atomtypes), self.Gs, self.int_types, dGs=self.dGs)
        self.train_dict = {self.pot.target: self.E_train,
            self.pot.target_forces: self.F_train,
            self.pot.error_weights: np.ones(len(self.atoms_train))}
        for i, t in enumerate(self.atomtypes):
            self.train_dict[self.pot.atom_indices[t]] = indices[i]
            if self.normalize_input == 'norm':
                norm_i = np.linalg.norm(ann_inputs[i], axis=1)
                self.train_dict[self.pot.atomic_contributions[t].input] = (
                    ann_inputs[i])/norm_i[:,np.newaxis]
                self.train_dict[
                    self.pot.atomic_contributions[t].derivatives_input
                    ] = np.einsum('ijkl,i->ijkl', ann_derivs[i], 1.0/norm_i)
            else:
                self.train_dict[self.pot.atomic_contributions[t].input] = (
                    ann_inputs[i]-self.Gs_norm1[t])/self.Gs_norm2[t]
                self.train_dict[
                    self.pot.atomic_contributions[t].derivatives_input
                    ] = np.einsum('ijkl,j->ijkl', ann_derivs[i], 1.0/self.Gs_norm2[t])

        if self.normalize_input == 'mean':
            for i, t in enumerate(self.atomtypes):
                self.Gs_norm1[t] = np.mean(ann_inputs[i], axis=0)
                # Small offset for numerical stability
                self.Gs_norm2[t] = np.std(ann_inputs[i], axis=0) + 1E-6
        elif self.normalize_input == 'min_max':
            for i, t in enumerate(self.atomtypes):
                self.Gs_norm1[t] =  np.min(ann_inputs[i], axis=0)
                # Small offset for numerical stability
                self.Gs_norm2[t] = (np.max(ann_inputs[i], axis=0) -
                    np.min(ann_inputs[i], axis=0) + 1E-6)

        # Start with large minimum loss value
        min_loss_value = 1E20
        for i in range(self.opt_restarts):
            # Reset weights to random initialization:
            if (i > 0 or self.reset_fit or
                    (self.opt_restarts == 1 and self.reset_fit)):
                self.session.run(tf.initializers.variables(self.pot.variables))
                # Do a few steps of ADAM optimization to start off:
                print('Starting ADAM optimization for %d epochs'%(
                    self.adam_maxiter))
                self.session.run(self.init_adam)
                for epoch in range(self.adam_maxiter):
                    self.session.run(self.train, self.train_dict)
                    if epoch%100 == 0:
                        loss_value, e_rmse, f_rmse = self.session.run(
                            [self.loss, self.pot.rmse, self.pot.rmse_forces],
                            self.train_dict)
                        # If ADAM converged to 10x convergence threshold, switch to
                        # scipy optimization
                        if (e_rmse/self.N_atoms < self.adam_thresh*0.001 and
                                f_rmse < self.adam_thresh*0.05):
                            print('Finished ADAM optimization after %d iterations.'%epoch,
                                'Total loss = %f, RMSE energy = %f, RMSE forces = %f.'%(
                                loss_value, e_rmse, f_rmse))
                            break

            # Use tensorflow optimizer interface to create a function that
            # returns both RMSEs given a vector of weights
            eval_rmse = self.optimizer._make_eval_func(
                [self.pot.rmse, self.pot.rmse_forces], self.session,
                self.train_dict, [])

            # Callback function that takes a vector of weights and checks if
            # the optimization is converged. Raises ConvergedNotAnError to stop
            # the scipy optimization if convergence criteria are met.
            def step_callback(packed_vars):
                rmse, rmse_forces = eval_rmse(packed_vars)
                # Check for convergence
                if rmse/self.N_atoms < self.e_tol and rmse_forces < self.f_tol:
                    # Copied straight from tensorflow optimizer interface.
                    # Typically the variables in the graph are updated at the
                    # end of the optimization. Since we are interupting the
                    # optimization (by raising an Exception) the variables have
                    # to be updated at this point.
                    var_vals = [packed_vars[packing_slice] for packing_slice in
                        self.optimizer._packing_slices]
                    self.session.run(
                        self.optimizer._var_updates, feed_dict=dict(zip(
                        self.optimizer._update_placeholders, var_vals))
                    )
                    raise ConvergedNotAnError()

            # Optimize weights using scipy.minimize
            try:
                self.optimizer.minimize(self.session, self.train_dict,
                    step_callback=step_callback)
            except ConvergedNotAnError:
                # This is actually the sucessful convergence of the optimization
                pass

            loss_value, e_rmse, f_rmse = self.session.run(
                [self.loss, self.pot.rmse, self.pot.rmse_forces],
                self.train_dict)
            print('Finished optimization %d/%d. '%(i+1, self.opt_restarts) +
                 'Total loss = %f, RMSE energy = %f, RMSE forces = %f.'%(
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
            '%f, RMSE force = %f.'%(e_rmse, f_rmse))

    def predict(self, atoms):
        int_types = [self.descriptor_set.type_dict[ti] for ti in
            atoms.get_chemical_symbols()]
        Gs, dGs = self.descriptor_set.eval_ase(atoms, derivatives=True)
        ann_inputs, indices, ann_derivs = calculate_bp_indices(
            len(self.atomtypes), [Gs], [int_types], dGs = [dGs])

        eval_dict = {self.pot.target: np.zeros(1),
            self.pot.target_forces: np.zeros((1, len(atoms), 3))}
        for i, t in enumerate(self.atomtypes):
            eval_dict[self.pot.atom_indices[t]] = indices[i]
            if self.normalize_input == 'norm':
                norm_i = np.linalg.norm(ann_inputs[i], axis=1)
                eval_dict[self.pot.atomic_contributions[t].input] = (
                    ann_inputs[i])/norm_i[:,np.newaxis]
                eval_dict[
                    self.pot.atomic_contributions[t].derivatives_input
                    ] = np.einsum('ijkl,i->ijkl', ann_derivs[i], 1.0/norm_i)
            else:
                eval_dict[self.pot.atomic_contributions[t].input] = (
                    ann_inputs[i]-self.Gs_norm1[t])/self.Gs_norm2[t]
                eval_dict[
                    self.pot.atomic_contributions[t].derivatives_input
                    ] = np.einsum('ijkl,j->ijkl', ann_derivs[i], 1.0/self.Gs_norm2[t])

        E = self.session.run(self.pot.E_predict, eval_dict)[0]
        F = self.session.run(self.pot.F_predict, eval_dict)[0]
        return E, F

    def get_params(self):
        params = {'normalize_input':self.normalize_input,
            'Gs_norm1':self.Gs_norm1, 'Gs_norm2':self.Gs_norm2,
            'model_dir':self.pot.saver.save(
                self.session, self.model_dir+'model.ckpt')}
        return params

    def set_params(self, **params):
        self.normalize_input = params['normalize_input']
        self.Gs_norm1 = params['Gs_norm1']
        self.Gs_norm2 = params['Gs_norm2']
        self.pot.saver.restore(self.session, params['model_dir'])

    def close(self):
        self.session.close()
