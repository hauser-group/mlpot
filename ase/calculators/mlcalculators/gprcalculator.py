from ase.calculators.mlcalculators.mlcalculator import MLCalculator
import numpy as np
from scipy.linalg import cho_solve, cholesky
import scipy.optimize as sp_opt
from scipy.optimize import minimize
import warnings

class GPRCalculator(MLCalculator):

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, C1=1.0, C2=1.0,
                 kernel=None,  opt_method='L-BFGS-B',
                 opt_restarts=0, normalize_y=False, **kwargs):
        MLCalculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms, C1, C2, **kwargs)

        self.kernel = kernel
        self.opt_method = opt_method
        self.opt_restarts = opt_restarts
        self.normalize_y = normalize_y

    def add_data(self, atoms):
        # If the trainings set is empty: setup the numpy arrays
        if not self.atoms_train:
            self.n_dim = 3*len(atoms)
            self.x_train = np.zeros((0, self.n_dim))
            self.E_train = np.zeros(0)
            self.F_train = np.zeros(0)
        # else: check if the new atoms object has the same length as previous
        else:
            if not 3*len(atoms) == self.n_dim:
                raise ValueError('New data does not have the same number of '
                    'atoms as previously added data.')

        # Call the super class routine after checking for empty trainings set!
        MLCalculator.add_data(self, atoms)
        self.x_train = np.append(
            self.x_train, atoms.get_positions().reshape((1,self.n_dim)), axis=0)
        self.E_train = np.append(self.E_train, atoms.get_potential_energy())
        self.F_train = np.append(self.F_train, atoms.get_forces().flatten())

    def fit(self):
        print('Fit called with %d geometries.'%len(self.atoms_train))
        self.n_samples = len(self.atoms_train)

        if self.normalize_y == 'mean':
            self.intercept = np.mean(self.E_train)
        elif self.normalize_y == 'min':
            self.intercept = np.min(self.E_train)
        elif self.normalize_y == False or self.normalize_y == None:
            self.intercept = 0.
        else:
            raise NotImplementedError('Unknown option: %s'%self.normalize_y)
        self._target_vector = np.concatenate(
            [self.E_train - self.intercept, -self.F_train.flatten()])

        if self.opt_restarts > 0:
            # TODO: Maybe it would be better to start from the same
            # initial_hyper_parameters (given at kernel initialization),
            # every time...

            # Lists to hold the results of the hyperparameter optimizations
            opt_hyper_parameter = []
            # List of values of the marginal log likelihood
            value = []
            for ii in range(self.opt_restarts):
                # First run: start from the current hyperparameters
                if ii == 0:
                    initial_hyper_parameters = self.kernel.theta
                # else: draw from log uniform distribution (drawing from
                # uniform but get and set hyper_parameter work with log values)
                else:
                    bounds = self.kernel.bounds
                    initial_hyper_parameters = np.zeros(len(bounds))
                    for bi, (lower_bound, upper_bound) in enumerate(bounds):
                        initial_hyper_parameters[bi] = np.random.uniform(
                            lower_bound, upper_bound, 1)
                print('Starting optimization %d/%d'%(ii+1, self.opt_restarts),
                    'with parameters: ', initial_hyper_parameters)
                try:
                    opt_x, val = self._opt_routine(initial_hyper_parameters)
                    opt_hyper_parameter.append(opt_x)
                    value.append(val)
                    print('Finished with value:', val,
                        ' and parameters:', opt_x)
                except np.linalg.LinAlgError as E:
                    print('Cholesky factorization failed for parameters:',
                        self.kernel.theta)
                    print(E)

            if len(value) == 0:
                raise ValueError('No successful optimization')
            # Find the optimum among all runs:
            min_idx = np.argmin(value)
            self.kernel.theta = opt_hyper_parameter[min_idx]

        k_mat = self.build_kernel_matrix()
        # Copy original k_mat (without regularization) for later calculation of
        # trainings error
        pure_k_mat = k_mat.copy()
        k_mat[:self.n_samples, :self.n_samples] += np.eye(
            self.n_samples)/self.C1
        k_mat[self.n_samples:, self.n_samples:] += np.eye(
            self.n_samples * self.n_dim)/self.C2

        self.L, alpha = self._cholesky(k_mat)
        self.alpha = alpha

        y = self.alpha.dot(pure_k_mat)
        E = y[:self.n_samples] + self.intercept
        F = -y[self.n_samples:]
        print('Fit finished. Final RMSE energy = %f, RMSE force = %f.'%(
            np.sqrt(np.mean((E - self.E_train)**2)),
            np.sqrt(np.mean((F - self.F_train)**2))))

    def _cholesky(self, kernel):
        """
        save routine to evaluate the cholesky factorization and weights
        :param kernel: kernel matrix
        :return: lower cholesky matrix, weights.
        """
        L = cholesky(kernel, lower=True)
        alpha = cho_solve((L, True), self._target_vector)
        return L, alpha

    def _opt_routine(self, initial_hyper_parameter):
        if self.opt_method in ['L-BFGS-B', 'SLSQP', 'TNC']:
            opt_obj = minimize(self._opt_fun, initial_hyper_parameter,
                method=self.opt_method, jac=True, bounds=self.kernel.bounds)
            opt_hyper_parameter = opt_obj.x
            value = opt_obj.fun
        else:
            raise NotImplementedError(
                'Method is not implemented or does not support the use of'
                'bounds use method=L-BFGS-B.')

        return opt_hyper_parameter, value

    def _opt_fun(self, hyper_parameter):
        """
        Function to optimize kernels hyper parameters
        :param hyper_parameter: new kernel hyper parameters
        :return: negative log marignal likelihood, derivative of the negative
                log marignal likelihood
        """
        self.kernel.theta = hyper_parameter
        log_marginal_likelihood, d_log_marginal_likelihood = (
            self.log_marginal_likelihood())

        return -log_marginal_likelihood, -d_log_marginal_likelihood

    def log_marginal_likelihood(self, derivative=False):
        """
        calculate the log marignal likelihood
        :return: log marinal likelihood,
            derivative of the log marignal likelihood w.r.t. the hyperparameters
        """
        # gives vale of log marginal likelihood with the gradient
        k_mat, k_grad = self.build_kernel_matrix(eval_gradient=True)
        k_mat[:self.n_samples, :self.n_samples] += np.eye(
            self.n_samples)/self.C1
        k_mat[self.n_samples:, self.n_samples:] += np.eye(
            self.n_samples*self.n_dim)/self.C2
        L, alpha = self._cholesky(k_mat)
        # Following Rasmussen Algorithm 2.1 the determinant in 2.30 can be
        # expressed as a sum over the Cholesky decomposition L
        log_mag_likelihood = (-0.5*self._target_vector.dot(alpha) -
            np.log(np.diag(L)).sum() - L.shape[0] / 2. * np.log(2 * np.pi))

        # summation inspired form scikit-learn Gaussian process regression
        temp = (np.multiply.outer(alpha, alpha) -
            cho_solve((L, True), np.eye(L.shape[0])))[:, :, np.newaxis]
        d_log_mag_likelihood = 0.5 * np.einsum("ijl,ijk->kl", temp, k_grad)
        d_log_mag_likelihood = d_log_mag_likelihood.sum(-1)

        return log_mag_likelihood, d_log_mag_likelihood

    def predict(self, atoms):
        # Prediction
        y = self.alpha.dot(self.build_kernel_matrix(X_star=atoms))
        E = y[0] + self.intercept
        F = -y[1:].reshape((-1,3))
        return E, F

    def get_params(self):
        return {'atoms_train':self.atoms_train, 'x_train':self.x_train,
            'alpha':self.alpha, 'intercept':self.intercept,
            'hyper_parameters':self.kernel.theta}

    def set_params(self, **params):
        self.atoms_train = params['atoms_train']
        self.x_train = params['x_train']
        self.n_dim = self.x_train.shape[1]
        self.alpha = params['alpha']
        self.intercept = params['intercept']
        self.kernel.theta = params['hyper_parameters']

    def build_kernel_matrix(self, X_star=None, eval_gradient=False):
        """Builds the kernel matrix K(X,X*) of the trainings_examples and
        X_star. If X_star==None the kernel of the trainings_examples with
        themselves K(X,X)."""
        if not X_star == None:
            x_star = np.array([X_star.get_positions().flatten()])
        else:
            x_star = self.x_train
        return self.kernel(self.x_train, x_star, dx=True, dy=True,
            eval_gradient=eval_gradient)
