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

    def fit(self, atoms_list):
        print('Fit called with %d geometries.'%len(atoms_list))
        # TODO check if all atoms objects have same size!
        self.n_dim = 3*len(atoms_list[0])
        self.x_train = np.zeros((len(atoms_list), self.n_dim))
        self.atoms_train = atoms_list
        self.E_train = np.zeros(len(atoms_list))
        self.F_train = np.zeros((len(atoms_list), self.n_dim))

        for i, atoms in enumerate(atoms_list):
            self.x_train[i,:] = atoms.get_positions().flatten()
            self.E_train[i] = atoms.get_potential_energy()
            self.F_train[i,:] = atoms.get_forces().flatten()

        self.n_samples = len(atoms_list)
        self.n_samples_force = len(atoms_list)
        self.F_train = self.F_train.flatten()#('F')

        if self.normalize_y == 'mean':
            self._intercept = np.mean(self.E_train)
        elif self.normalize_y == 'min':
            self._intercept = np.min(self.E_train)
        elif self.normalize_y == False or self.normalize_y == None:
            self._intercept = 0.
        else:
            raise NotImplementedError('Unknown option: %s'%self.normalize_y)
        self._target_vector = np.concatenate(
            [self.E_train - self._intercept, -self.F_train])

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
                    initial_hyper_parameters = self.get_hyper_parameter()
                # else: draw from log uniform distribution (drawing from
                # uniform but get and set hyper_parameter work with log values)
                else:
                    bounds = self.kernel.bounds
                    initial_hyper_parameters = np.zeros(len(bounds))
                    for bi, element in enumerate(bounds):
                        initial_hyper_parameters[bi] = np.random.uniform(
                            element[0], element[1], 1)
                print('Starting optimization %d/%d'%(ii+1, self.opt_restarts),
                    'with parameters: ', initial_hyper_parameters)
                try:
                    opt = self._opt_routine(initial_hyper_parameters)
                    opt_hyper_parameter.append(opt[0])
                    value.append(opt[1])
                    print('Finished with value:', opt[1],
                        ' and parameters:', opt[0])
                except np.linalg.LinAlgError:
                    print('Cholesky factorization failed')

            if len(value) == 0:
                raise ValueError('No successful optimization')
            # Find the optimum among all runs:
            min_idx = np.argmin(value)
            self.set_hyper_parameter(opt_hyper_parameter[min_idx])

        k_mat = self.kernel(
            self.atoms_train, self.atoms_train, dx=True, dy=True)
        k_mat[:self.n_samples, :self.n_samples] += np.eye(
            self.n_samples)/self.C1
        k_mat[self.n_samples:, self.n_samples:] += np.eye(
            self.n_samples_force * self.n_dim)/self.C2

        self.L, alpha = self._cholesky(k_mat)
        self.alpha = alpha
        self._alpha = alpha[:self.n_samples]
        self._beta = alpha[self.n_samples:].reshape(self.n_dim, -1).T

    def _cholesky(self, kernel):
        """
        save routine to evaluate the cholesky factorization and weights
        :param kernel: kernel matrix
        :return: lower cholesky matrix, weights.
        """
        L = cholesky(kernel, lower=True)
        alpha = cho_solve((L, True), self._target_vector)
        return L, alpha

    def get_hyper_parameter(self):
        return self.kernel.theta

    def set_hyper_parameter(self, hyper_parameter):
        self.kernel.theta = hyper_parameter

    def get_bounds(self):
        return self.kernel.bounds

    def optimize(self, hyper_parameter):
        """
        Function to optimize kernels hyper parameters
        :param hyper_parameter: new kernel hyper parameters
        :return: negative log marignal likelihood, derivative of the negative
                log marignal likelihood
        """
        self.set_hyper_parameter(hyper_parameter)
        log_marginal_likelihood, d_log_marginal_likelihood = self.log_marginal_likelihood(
            derivative=self._opt_flag)

        return -log_marginal_likelihood, -d_log_marginal_likelihood

    def log_marginal_likelihood(self, derivative=False):
        """
        calculate the log marignal likelihood
        :param derivative: determines if the derivative to the log marignal likelihood should be evaluated. default False
        :return: log marinal likelihood, derivative of the log marignal likelihood
        """
        # gives vale of log marginal likelihood with the gradient
        k_mat, k_grad = self.kernel(self.atoms_train, self.atoms_train,
            dx=True, dy=True, eval_gradient=True)
        k_mat[:self.n_samples, :self.n_samples] += np.eye(self.n_samples)/self.C1
        k_mat[self.n_samples:, self.n_samples:] += np.eye(self.n_samples_force*self.n_dim)/self.C2
        L, alpha = self._cholesky(k_mat)
        # Following Rasmussen Algorithm 2.1 the determinant in 2.30 can be
        # expressed as a sum over the Cholesky decomposition L
        log_mag_likelihood = -0.5*self._target_vector.dot(alpha) - np.log(np.diag(L)).sum() - L.shape[0] / 2. * np.log(2 * np.pi)

        if not derivative:
            return log_mag_likelihood
        # summation inspired form scikit-learn Gaussian process regression
        temp = (np.multiply.outer(alpha, alpha) - cho_solve((L, True), np.eye(L.shape[0])))[:, :, np.newaxis]
        d_log_mag_likelihood = 0.5 * np.einsum("ijl,ijk->kl", temp, k_grad)
        d_log_mag_likelihood = d_log_mag_likelihood.sum(-1)

        return log_mag_likelihood, d_log_mag_likelihood

    def _opt_routine(self, initial_hyper_parameter):
        if self.opt_method == 'L-BFGS-B':
            self._opt_flag = True
            opt_obj = minimize(self.optimize, initial_hyper_parameter,
                method='L-BFGS-B', jac=True, bounds=self.get_bounds())
            opt_hyper_parameter = opt_obj.x
            value = opt_obj.fun
        elif self.opt_method == 'LBFGS_B':
            self._opt_flag = True
            opt_hyper_parameter, value, opt_dict = sp_opt.fmin_l_bfgs_b(
                self.optimize, initial_hyper_parameter,
                bounds=self.get_bounds())
            if opt_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the state: %s" % opt_dict)
        elif self.opt_method == 'BFGS':
            raise NotImplementedError('Implementation is not finished.')
            # TODO return function value
            # TODO implementation
            self._opt_flag = False
            opt_hyper_parameter = sp_opt.fmin_bfgs(self.optimize, initial_hyper_parameter)
            value = 0
        else:
            raise NotImplementedError('Method is not implemented use method=LBFGS_B.')

        return opt_hyper_parameter, value

    def predict(self, atoms):
        # Prediction
        y = self.alpha.dot(self.kernel(self.atoms_train, [atoms], dx=True, dy=True))
        E = y[0] + self._intercept
        F = -y[1:].reshape((-1,3))
        return E, F

    def get_params(self):
        return {'x_train':self.x_train, 'atoms_train':self.atoms_train,
            'alpha':self.alpha, '_alpha':self._alpha, '_beta':self._beta,
            'intercept':self._intercept,
            'hyper_parameters':self.get_hyper_parameter()}

    def set_params(self, **params):
        self.x_train = params['x_train']
        self.atoms_train = params['atoms_train']
        self.n_dim = self.x_train.shape[1]
        self.alpha = params['alpha']
        self._alpha = params['_alpha']
        self._beta = params['_beta']
        self._intercept = params['intercept']
        self.set_hyper_parameter(params['hyper_parameters'])

def create_mat(kernel, x1, x2, dx_max=0, dy_max=0, eval_gradient=False):
    """
    creates the kernel matrix with respect to the derivatives.
    :param kernel: given kernel like RBF
    :param x1: training points shape (n_samples, n_features)
    :param x2: training or prediction points (n_samples, n_features)
    :param dx_max: maximum derivative in x1_prime
    :param dy_max: maximum derivative in x2_prime
    :param eval_gradient: flag if kernels derivative have to be evaluated. default False
    :return: kernel matrix, derivative of the kernel matrix
    """
    # creates the kernel matrix
    # if x1_prime is None then no derivative elements are calculated.
    # derivative elements are given in the manner of [dx1, dx2, dx3, ...., dxn]
    n, d = x1.shape
    m, f = x2.shape
    if not eval_gradient:
        kernel_mat = np.zeros([n * (1 + d), m * (1 + f)])
        for ii in range(dx_max + 1):
            for jj in range(dy_max + 1):
                kernel_mat[n * ii:n * (ii + 1), m * jj:m * (jj + 1)] = kernel(
                    x1, x2, dx=ii, dy=jj, eval_gradient=False)
        return kernel_mat
    else:
        num_theta = len(kernel.theta)
        kernel_derivative = np.zeros([n * (1 + d), m * (1 + f), num_theta])
        kernel_mat = np.zeros([n * (1 + d), m * (1 + f)])
        for ii in range(dx_max + 1):
            for jj in range(dy_max + 1):
                k_mat, deriv_mat = kernel(x1, x2, dx=ii, dy=jj, eval_gradient=True)

                kernel_mat[n * ii:n * (ii + 1), m * jj:m * (jj + 1)] = k_mat
                kernel_derivative[n * ii:n * (ii + 1), m * jj:m * (jj + 1), :] = deriv_mat
    return kernel_mat, kernel_derivative
