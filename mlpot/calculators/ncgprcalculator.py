from mlpot.calculators.mlcalculator import MLCalculator
from mlpot.calculators.gprcalculator import GPRCalculator
import numpy as np


class NCGPRCalculator(GPRCalculator):
    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, input_transform=None, **kwargs):
        GPRCalculator.__init__(self, restart, ignore_bad_restart_file, label,
                               atoms, **kwargs)
        self.input_transform = input_transform

    def add_data(self, atoms):
        # If the trainings set is empty: setup the numpy arrays
        if not self.atoms_train:
            self.n_dim = 3*len(atoms)
            self.E_train = np.zeros(0)
            self.F_train = np.zeros(0)
            self.q_train = []
            self.dq_train = []
        # else: check if the new atoms object has the same length as previous
        else:
            if not 3*len(atoms) == self.n_dim:
                raise ValueError('New data does not have the same number of '
                                 'atoms as previously added data.')

        # Call the super class routine after checking for empty trainings set!
        MLCalculator.add_data(self, atoms)
        # Call forces first in case forces and energy are calculated at the
        # same time by the calculator
        if self.mean_model is None:
            F = atoms.get_forces().flatten()
            E = atoms.get_potential_energy()
        else:
            F = (atoms.get_forces().flatten()
                 - self.mean_model.get_forces(atoms=atoms).flatten())
            E = (atoms.get_potential_energy()
                 - self.mean_model.get_potential_energy(atoms=atoms))
        self.E_train = np.append(self.E_train, E)
        self.F_train = np.append(self.F_train, F)

        q, dq = self._transform_input(atoms)
        self.q_train.append(q)
        self.dq_train.append(dq)

    def delete_data(self, indices=None):
        if indices is None:
            indices = slice(len(self.atoms_train))
        del self.atoms_train[indices]
        del self.q_train[indices]
        del self.dq_train[indices]
        self.E_train = np.delete(self.E_train, indices, 0)
        self.F_train = np.delete(self.F_train.reshape(-1, self.n_dim),
                                 indices, 0).reshape(-1)

    def _transform_input(self, atoms):
        return self.input_transform(atoms)

    def _normalize_input(self, x):
        return x

    def get_params(self):
        return {'atoms_train': self.atoms_train,
                'q_train': self.q_train,
                'dq_train': self.dq_train,
                'alpha': self.alpha,
                'L': self.L,
                'intercept': self.intercept,
                'mean_model': self.mean_model,
                'hyperparameters': self.kernel.theta}

    def set_params(self, **params):
        self.atoms_train = params['atoms_train']
        self.q_train = params['q_train']
        self.dq_train = params['dq_train']
        self.n_dim = 3*len(self.atoms_train[0])
        self.alpha = params['alpha']
        self.L = params['L']
        self.intercept = params['intercept']
        self.mean_model = params['mean_model']
        self.kernel.theta = params['hyperparameters']

    def build_kernel_matrix(self, X_star=None, eval_gradient=False):
        n = len(self.q_train)
        q_X = np.array(self.q_train)
        dq_X = np.array(self.dq_train)
        n_qs = q_X.shape[1]

        if X_star is None:
            m = n
            q_Y = q_X
            dq_Y = dq_X
        else:
            m = 1
            q_Y = np.atleast_2d(X_star[0])
            # Ugly but a quick fix for np.atleast_3d appending the new 3rd
            # dimension to the end
            dq_Y = np.array([X_star[1]])

        kernel_mat = np.zeros((n*(1+self.n_dim), m*(1+self.n_dim)))
        if eval_gradient:
            # kernel in q coordinates
            K_q, dK_q = self.kernel(q_X, q_Y, dx=True, dy=True,
                                    eval_gradient=True)
        else:
            K_q = self.kernel(q_X, q_Y, dx=True, dy=True)
        # no transformation needed for function values
        kernel_mat[:n, :m] = K_q[:n, :m]
        # i and j are used as index for the cartesian coordinates
        # k and l used as index for the q coordinates
        # K_q[n:, :m] size n*n_qs x m reshape to n x n_qs x m
        # dq_X size n x n_qs x n_dim
        kernel_mat[n:, :m] = np.einsum(
                'nkm,nki->nim', K_q[n:, :m].reshape(n, n_qs, m), dq_X
            ).reshape(n*self.n_dim, m)
        # K_q[:n, m:] size n x m*n_qs reshape to n x m x n_qs
        # dq_Y size m x n_qs x n_dim
        kernel_mat[:n, m:] = np.einsum(
                'nml,mlj->nmj', K_q[:n, m:].reshape(n, m, n_qs), dq_Y
            ).reshape(n, m*self.n_dim)
        # K_q[n:, m:] size n*n_qs x m*n_qs reshape to n x n_qs x m x n_qs
        kernel_mat[n:, m:] = np.einsum(
                'nki,nkml,mlj->nimj',
                dq_X, K_q[n:, m:].reshape(n, n_qs, m, n_qs),
                dq_Y, optimize=True).reshape(n*self.n_dim, m*self.n_dim)

        if eval_gradient:
            n_t = len(self.kernel.theta)
            kernel_grad = np.zeros(
                (n*(1+self.n_dim), m*(1+self.n_dim), n_t))
            kernel_grad[:n, :m, :] = dK_q[:n, :m, :]
            # Works just the same way as shown above with the additional
            # index t for the hyperparameters
            kernel_grad[n:, :m, :] = np.einsum(
                    'nkmt,nki->nimt',
                    dK_q[n:, :m, :].reshape(n, n_qs, m, n_t), dq_X
                ).reshape(n*self.n_dim, m, n_t)
            kernel_grad[:n, m:, :] = np.einsum(
                    'nmlt,mlj->nmjt',
                    dK_q[:n, m:, :].reshape(n, m, n_qs, n_t), dq_Y
                ).reshape(n, m*self.n_dim, n_t)
            kernel_grad[n:, m:, :] = np.einsum(
                    'nki,nkmlt,mlj->nimjt', dq_X,
                    dK_q[n:, m:, :].reshape(n, n_qs, m, n_qs, n_t),
                    dq_Y, optimize=True
                ).reshape(n*self.n_dim, m*self.n_dim, n_t)
            return kernel_mat, kernel_grad
        else:
            return kernel_mat

    def build_kernel_diagonal(self, X_star):
        """Evaluates the diagonal of the kernel matrix which is needed for
        the uncertainty prediction. TODO: reaccess if there is a better way to
        do this.
        """
        q_star = np.atleast_2d(X_star[0])
        # Ugly but a quick fix for np.atleast_3d appending the new 3rd
        # dimension to the end
        if np.ndim(X_star[1]) == 2:
            dq_star = np.array([X_star[1]])
        elif np.ndim(X_star[1]) == 3:
            dq_star = np.array(X_star[1])
        n = q_star.shape[0]
        n_qs = q_star.shape[1]

        K_q = self.kernel(q_star, q_star, dx=True, dy=True)
        kernel_diag = np.zeros((n*(1+self.n_dim)))
        kernel_diag[:n] = K_q.diagonal()[:n]
        kernel_diag[n:] = np.einsum(
                'nki,nknl,nli->ni',
                dq_star, K_q[n:, n:].reshape(n, n_qs, n, n_qs),
                dq_star, optimize=True).flatten()
        return kernel_diag
