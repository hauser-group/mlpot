from mlpot.calculators.mlcalculator import MLCalculator
from mlpot.calculators.gprcalculator import GPRCalculator
import numpy as np
import copy


class GAPCalculator(GPRCalculator):

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, descriptor_set=None,
                 normalize_input=False, **kwargs):
        GPRCalculator.__init__(self, restart, ignore_bad_restart_file, label,
                               atoms, **kwargs)

        if descriptor_set is None:
            raise NotImplementedError('For now a descriptor set has to be ' +
                                      'passed to the constructor as no ' +
                                      'default set is implemented')
        else:
            self.descriptor_set = descriptor_set

        self.atomtypes = self.descriptor_set.atomtypes
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

    def add_data(self, atoms):
        """The data structure for the descriptors is somewhat complicated.
        Needs a more extensive description!!!"""
        # If the trainings set is empty: setup the numpy arrays
        if not self.atoms_train:
            self.n_dim = 3*len(atoms)
            self.E_train = np.zeros(0)
            self.F_train = np.zeros(0)
            self.Gs = {t: [] for t in self.atomtypes}
            self.dGs = {t: [] for t in self.atomtypes}
        # else: check if the new atoms object has the same length as previous
        else:
            if not 3*len(atoms) == self.n_dim:
                raise ValueError('New data does not have the same number of'
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

        Gs_by_type, dGs_by_type = self._transform_input(atoms)
        for t in self.atomtypes:
            self.Gs[t].extend(Gs_by_type[t])
            self.dGs[t].extend(dGs_by_type[t])

    def delete_data(self, indices=None):
        if indices is None:
            indices = slice(len(self.atoms_train))
        del self.atoms_train[indices]
        for t in self.atomtypes:
            del self.Gs[t][indices]
            del self.dGs[t][indices]
        self.E_train = np.delete(self.E_train, indices, 0)
        self.F_train = np.delete(self.F_train.reshape(-1, self.n_dim),
                                 indices, 0).reshape(-1)

    def _transform_input(self, atoms):
        # TODO: extend to allow transforming a list of atoms
        types = atoms.get_chemical_symbols()
        Gs, dGs = self.descriptor_set.eval_ase(atoms, derivatives=True)
        Gs_by_type = {t: [] for t in self.atomtypes}
        dGs_by_type = {t: [] for t in self.atomtypes}
        for Gi, dGi, ti in zip(Gs, dGs, types):
            Gs_by_type[ti].append(Gi)
            dGs_by_type[ti].append(dGi.reshape((-1, self.n_dim)))
        for t in self.atomtypes:
            Gs_by_type[t] = [np.array(Gs_by_type[t])]
            dGs_by_type[t] = [np.array(dGs_by_type[t])]
        return Gs_by_type, dGs_by_type

    def _normalize_input(self, x):
        """Copies the input an returns a normalized version"""
        Gs_norm = copy.deepcopy(x[0])
        dGs_norm = copy.deepcopy(x[1])
        # Iterate over atomtypes
        for t in self.atomtypes:
            # Iterate over number of data points
            for i in range(len(Gs_norm[t])):
                if (self.normalize_input == 'mean' or
                        self.normalize_input == 'min_max'):
                    Gs_norm[t][i] = (
                        Gs_norm[t][i]-self.Gs_norm1[t])/self.Gs_norm2[t]
                    dGs_norm[t][i] = np.einsum('ijk,j->ijk', dGs_norm[t][i],
                                               1.0/self.Gs_norm2[t])
                elif self.normalize_input == 'norm':
                    norm_i = np.linalg.norm(Gs_norm[t][i], axis=1)
                    Gs_norm[t][i] = (Gs_norm[t][i]/norm_i[:, np.newaxis])
                    dGs_norm[t][i] = np.einsum('ijk,i->ijk', dGs_norm[t][i],
                                               1.0/norm_i)
                # Reshaping can only be done here as the separate descriptor
                # dimension is needed for normalization
                dGs_norm[t][i] = dGs_norm[t][i].reshape((-1, self.n_dim))
        return Gs_norm, dGs_norm

    def fit(self):
        if self.normalize_input == 'mean':
            for i, t in enumerate(self.atomtypes):
                Gs_t = np.array(self.Gs[t])
                self.Gs_norm1[t] = np.mean(Gs_t, axis=(0, 1))
                # Small offset for numerical stability
                self.Gs_norm2[t] = np.std(Gs_t, axis=(0, 1)) + 1E-6
        elif self.normalize_input == 'min_max':
            for i, t in enumerate(self.atomtypes):
                Gs_t = np.array(self.Gs[t])
                self.Gs_norm1[t] = np.min(Gs_t, axis=(0, 1))
                # Small offset for numerical stability
                self.Gs_norm2[t] = (np.max(Gs_t, axis=(0, 1)) -
                                    np.min(Gs_t, axis=(0, 1)) + 1E-6)

        self.Gs_norm, self.dGs_norm = self._normalize_input(
            (self.Gs, self.dGs))
        GPRCalculator.fit(self)

    def get_params(self):
        return {'atoms_train': self.atoms_train,
                'normalize_input': self.normalize_input,
                'Gs': self.Gs,
                'dGs': self.dGs,
                'Gs_norm': self.Gs_norm,
                'dGs_norm': self.dGs_norm,
                'Gs_norm1': self.Gs_norm1,
                'Gs_norm2': self.Gs_norm2,
                'alpha': self.alpha,
                'L': self.L,
                'intercept': self.intercept,
                'hyper_parameters': self.kernel.theta}

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
        self.alpha = params['alpha']
        self.L = params['L']
        self.intercept = params['intercept']
        self.kernel.theta = params['hyper_parameters']

    def build_kernel_matrix(self, X_star=None, eval_gradient=False):
        """Builds the kernel matrix K(X,X*) of the trainings_examples and
        X_star. If X_star==None the kernel of the trainings_examples with
        themselves K(X,X)."""
        N = len(self.atoms_train)
        Gs_X = self.Gs_norm
        dGs_X = self.dGs_norm

        if X_star is None:
            M = N
            Gs_Y = Gs_X
            dGs_Y = dGs_X
        else:
            M = 1
            Gs_Y = X_star[0]
            dGs_Y = X_star[1]

        if eval_gradient:
            kernel_grad = np.zeros(
                (N*(1+self.n_dim), M*(1+self.n_dim), len(self.kernel.theta)))
        kernel_mat = np.zeros((N*(1+self.n_dim), M*(1+self.n_dim)))
        for t in self.atomtypes:
            for i, (Gsi_t, dGsi_t) in enumerate(zip(Gs_X[t], dGs_X[t])):
                for j, (Gsj_t, dGsj_t) in enumerate(zip(Gs_Y[t], dGs_Y[t])):
                    n_t = Gsi_t.shape[0]
                    m_t = Gsj_t.shape[0]
                    # Index range for the derivatives of geometry i and j
                    # w.r.t. the cartesian coordinates:
                    di = slice(N+i*self.n_dim, N+(i+1)*self.n_dim, 1)
                    dj = slice(M+j*self.n_dim, M+(j+1)*self.n_dim, 1)
                    # Calculate the kernel matrix of between atoms of type
                    # t in geometry i and j. Shape
                    # (n_t+n_t*num_Gs_t, m_t+m_t*num_Gs_t)
                    if eval_gradient:
                        K, dK = self.kernel(Gsi_t, Gsj_t, dx=True, dy=True,
                                            eval_gradient=True)
                    else:
                        K = self.kernel(Gsi_t, Gsj_t, dx=True, dy=True)
                    # Sum over atoms in n_t and m_t
                    kernel_mat[i, j] += np.sum(K[:n_t, :m_t])
                    # dGsi original shape = (n_t, num_Gs_t, num_atoms_i, 3)
                    kernel_mat[di, j] += np.einsum(
                        'ij,il->l', K[n_t:, :m_t], dGsi_t)
                    kernel_mat[i, dj] += np.einsum(
                        'ij,jl->l', K[:n_t, m_t:], dGsj_t)
                    kernel_mat[di, dj] += (
                        dGsi_t.T.dot(K[n_t:, m_t:]).dot(dGsj_t))
                    if eval_gradient:
                        kernel_grad[i, j, :] += np.sum(
                            dK[:n_t, :m_t, :], axis=(0, 1))
                        kernel_grad[di, j, :] += np.einsum(
                            'ijk,il->lk', dK[n_t:, :m_t, :], dGsi_t)
                        kernel_grad[i, dj, :] += np.einsum(
                            'ijk,jl->lk', dK[:n_t, m_t:, :], dGsj_t)
                        # Numpys einsum is not optimized for three arguments by
                        # default. The fix for now is to use the optimize flag.
                        # Maybe check if this can be done using tensordot or
                        # something similar and compare performance.
                        kernel_grad[di, dj, :] += np.einsum(
                            'il,ijk,jm->lmk', dGsi_t,
                            dK[n_t:, m_t:, :], dGsj_t, optimize=True)
        if eval_gradient:
            return kernel_mat, kernel_grad
        else:
            return kernel_mat

    def build_kernel_diagonal(self, X_star):
        """Evaluates the diagonal of the kernel matrix which can be done
        significantly faster than evaluating the whole matrix and is needed for
        the uncertainty prediction.
        """
        Gs_X = X_star[0]
        dGs_X = X_star[1]
        n = len(Gs_X[self.atomtypes[0]])

        kernel_diag = np.zeros(n*(1+self.n_dim))
        for t in self.atomtypes:
            for i, (Gsi_t, dGsi_t) in enumerate(zip(Gs_X[t], dGs_X[t])):
                n_t = Gsi_t.shape[0]
                di = slice(n+i*self.n_dim, n+(i+1)*self.n_dim, 1)

                K = self.kernel(Gsi_t, Gsi_t, dx=True, dy=True)
                kernel_diag[i] += np.sum(K[:n_t, :n_t])
                # n_dim x n * n_dim x n_dim * n_dim x n -> n x 1
                kernel_diag[di] += np.einsum(
                    'ji,jk,ki->i', dGsi_t, K[n_t:, n_t:], dGsi_t,
                    optimize=True)
        return kernel_diag
