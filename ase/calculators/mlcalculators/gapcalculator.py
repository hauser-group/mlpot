from ase.calculators.mlcalculators.mlcalculator import MLCalculator
from ase.calculators.mlcalculators.gprcalculator import GPRCalculator
import numpy as np

class GAPCalculator(GPRCalculator):

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, C1=1.0, C2=1.0,
                 descriptor_set=None, kernel=None, opt_method='L-BFGS-B',
                 opt_restarts=0, normalize_y=False, normalize_input=False,
                 **kwargs):
        GPRCalculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms, C1, C2, kernel=kernel,
                            opt_method=opt_method, opt_restarts=opt_restarts,
                            normalize_y=normalize_y, **kwargs)

        if descriptor_set == None:
            raise NotImplementedError('For now a descriptor set has to be' +
                ' passed to the constructor as no default set is implemented')
        else:
            self.descriptor_set = descriptor_set

        self.atomtypes = self.descriptor_set.atomtypes
        if not normalize_input in ['mean', 'min_max', 'norm', False]:
            raise NotImplementedError(
                'Unknown input normalization %s'%normalize_input)
        self.normalize_input = normalize_input
        # Depending on the type of input normalization these dicts hold
        # different values, for example the mean and the std for "mean" norm
        # or the minimum and the maximum-minumum difference for "min_max"
        self.Gs_norm1 = {}
        self.Gs_norm2 = {}
        for t in self.atomtypes:
            self.Gs_norm1[t] = np.zeros(self.descriptor_set.num_Gs[
                self.descriptor_set.type_dict[t]])
            self.Gs_norm2[t] = np.ones(self.descriptor_set.num_Gs[
                self.descriptor_set.type_dict[t]])

    def add_data(self, atoms):
        # If the trainings set is empty: setup the numpy arrays
        if not self.atoms_train:
            self.n_dim = 3*len(atoms)
            self.E_train = np.zeros(0)
            self.F_train = np.zeros(0)
            self.Gs = {t:[] for t in self.atomtypes}
            self.dGs = {t:[] for t in self.atomtypes}
        # else: check if the new atoms object has the same length as previous
        else:
            if not 3*len(atoms) == self.n_dim:
                raise ValueError('New data does not have the same number of'
                    'atoms as previously added data.')

        # Call the super class routine after checking for empty trainings set!
        MLCalculator.add_data(self, atoms)
        self.E_train = np.append(self.E_train, atoms.get_potential_energy())
        self.F_train = np.append(self.F_train, atoms.get_forces().flatten())

        types = atoms.get_chemical_symbols()
        Gs, dGs = self.descriptor_set.eval_ase(atoms, derivatives=True)
        Gs_by_type = {t:[] for t in self.atomtypes}
        dGs_by_type = {t:[] for t in self.atomtypes}
        for Gi, dGi, ti in zip(Gs, dGs, types):
            Gs_by_type[ti].append(Gi)
            dGs_by_type[ti].append(dGi.reshape((-1, self.n_dim)))
        for t in self.atomtypes:
            self.Gs[t].append(np.array(Gs_by_type[t]))
            self.dGs[t].append(np.array(dGs_by_type[t]))

    def fit(self):
        if self.normalize_input == 'mean':
            for i, t in enumerate(self.atomtypes):
                Gs_t = np.array(self.Gs[t])
                print(Gs_t.shape)
                self.Gs_norm1[t] = np.mean(Gs_t, axis=(0,1))
                # Small offset for numerical stability
                self.Gs_norm2[t] = np.std(Gs_t, axis=(0,1)) + 1E-6
        elif self.normalize_input == 'min_max':
            for i, t in enumerate(self.atomtypes):
                Gs_t = np.array(self.Gs[t])
                print(Gs_t.shape)
                self.Gs_norm1[t] = np.min(Gs_t, axis=(0,1))
                # Small offset for numerical stability
                self.Gs_norm2[t] = (
                    np.max(Gs_t, axis=(0,1)) - np.min(Gs_t, axis=(0,1)) + 1E-6)

        GPRCalculator.fit(self)

    def get_params(self):
        return {'atoms_train':self.atoms_train, 'Gs':self.Gs, 'dGs':self.dGs,
            'alpha':self.alpha, 'intercept':self.intercept,
            'hyper_parameters':self.kernel.theta}

    def set_params(self, **params):
        self.atoms_train = params['atoms_train']
        self.Gs = params['Gs']
        self.dGs = params['dGs']
        self.n_dim = 3*len(self.atoms_train[0])
        self.alpha = params['alpha']
        self.intercept = params['intercept']
        self.kernel.theta = params['hyper_parameters']

    def build_kernel_matrix(self, X_star=None, eval_gradient=False):
        """Builds the kernel matrix K(X,X*) of the trainings_examples and
        X_star. If X_star==None the kernel of the trainings_examples with
        themselves K(X,X)."""
        N = len(self.atoms_train)
        Gs_X = {t:[] for t in self.atomtypes}
        dGs_X = {t:[] for t in self.atomtypes}
        for t in self.atomtypes:
            for Gsi, dGsi in zip(self.Gs[t], self.dGs[t]):
                if (self.normalize_input == 'mean' or
                        self.normalize_input == 'min_max'):
                    Gs_X[t].append((Gsi-self.Gs_norm1[t])/self.Gs_norm2[t])
                    dGs_X[t].append(np.einsum('ijk,j->ijk', dGsi,
                        1.0/self.Gs_norm2[t]).reshape((-1, self.n_dim)))
                elif self.normalize_input == 'norm':
                    norm_i = np.linalg.norm(Gsi, axis=1)
                    Gs_X[t].append(Gsi/norm_i[:,np.newaxis])
                    dGs_X[t].append(np.einsum('ijk,i->ijk', dGsi,
                        1.0/norm_i).reshape((-1, self.n_dim)))
                elif not self.normalize_input:
                    Gs_X[t].append(Gsi)
                    dGs_X[t].append(dGsi.reshape((-1, self.n_dim)))
        if not X_star == None:
            M = 1
            types = X_star.get_chemical_symbols()
            Gs, dGs = self.descriptor_set.eval_ase(X_star, derivatives=True)
            Gs_Y = {t:[] for t in self.atomtypes}
            dGs_Y = {t:[] for t in self.atomtypes}
            Gs_by_type = {t:[] for t in self.atomtypes}
            dGs_by_type = {t:[] for t in self.atomtypes}
            for Gi, dGi, ti in zip(Gs, dGs, types):
                Gs_by_type[ti].append(Gi)
                dGs_by_type[ti].append(dGi.reshape((-1, self.n_dim)))
            for t in self.atomtypes:
                Gsi = np.array(Gs_by_type[t])
                dGsi = np.array(dGs_by_type[t])
                if (self.normalize_input == 'mean' or
                        self.normalize_input == 'min_max'):
                    Gs_Y[t].append((Gsi-self.Gs_norm1[t])/self.Gs_norm2[t])
                    dGs_Y[t].append(np.einsum('ijk,j->ijk', dGsi,
                        1.0/self.Gs_norm2[t]).reshape((-1, self.n_dim)))
                elif self.normalize_input == 'norm':
                    norm_i = np.linalg.norm(Gsi, axis=1)
                    Gs_Y[t].append(Gsi/norm_i[:,np.newaxis])
                    dGs_Y[t].append(np.einsum('ijk,i->ijk', dGsi,
                        1.0/norm_i).reshape((-1, self.n_dim)))
                elif not self.normalize_input:
                    Gs_Y[t].append(Gsi)
                    dGs_Y[t].append(dGsi.reshape((-1, self.n_dim)))
        else:
            M = N
            Gs_Y = Gs_X
            dGs_Y = dGs_X

        if eval_gradient:
            kernel_grad = np.zeros(
                (N*(1+self.n_dim), M*(1+self.n_dim),len(self.kernel.theta)))
        kernel_mat = np.zeros((N*(1+self.n_dim), M*(1+self.n_dim)))
        for t in self.atomtypes:
            for i, (Gsi_t, dGsi_t) in enumerate(zip(Gs_X[t], dGs_X[t])):
                for j, (Gsj_t, dGsj_t) in enumerate(zip(Gs_Y[t], dGs_Y[t])):

                    # TODO: Not a great point to do the normalization...
                    #Gsi_t = (Gsi[t]-self.Gs_norm1[t])/self.Gs_norm2[t]
                    #Gsj_t = (Gsj[t]-self.Gs_norm1[t])/self.Gs_norm2[t]
                    #dGsi_t = np.einsum('ijkl,j->ijkl', dGsi[t],
                    #    1.0/self.Gs_norm2[t]).reshape((-1, self.n_dim))
                    #dGsj_t = np.einsum('ijkl,j->ijkl', dGsj[t],
                    #    1.0/self.Gs_norm2[t]).reshape((-1, self.n_dim))
                    #dGsi_t = dGsi_t.reshape((-1, self.n_dim))
                    #dGsj_t = dGsj_t.reshape((-1, self.n_dim))

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

                    kernel_mat[di, j] += np.einsum('ij,il->l',
                        K[n_t:, :m_t], dGsi_t)
                    kernel_mat[i, dj] += np.einsum('ij,jl->l',
                        K[:n_t, m_t:], dGsj_t)
                    kernel_mat[di, dj] += (dGsi_t.T.dot(K[n_t:, m_t:]).dot(
                        dGsj_t))
                    if eval_gradient:
                        kernel_grad[i,j,:] += np.sum(dK[:n_t,:m_t,:])
                        kernel_grad[di,j,:] += np.einsum('ijk,il->lk',
                            dK[n_t:,:m_t,:], dGsi_t)
                        kernel_grad[i,dj,:] += np.einsum('ijk,jl->lk',
                            dK[:n_t,m_t:,:], dGsj_t)
                        kernel_grad[di,dj,:] += np.einsum('il,ijk,jm->lmk',
                            dGsi_t, dK[n_t:,m_t:,:], dGsj_t)
        if eval_gradient:
            return kernel_mat, kernel_grad
        else:
            return kernel_mat
