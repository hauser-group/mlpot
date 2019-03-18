from sklearn.gaussian_process.kernels import RBF as sk_RBF
from sklearn.gaussian_process.kernels import Sum as sk_Sum
from sklearn.gaussian_process.kernels import ConstantKernel as sk_ConstantKernel
from sklearn.gaussian_process.kernels import _check_length_scale
from scipy.spatial.distance import cdist
import numpy as np

class KlemensInterface(object):
    def __init__(self, kernel):
        self.kernel = kernel

    @property
    def bounds(self):
        return self.kernel.bounds

    @property
    def theta(self):
        return self.kernel.theta

    @theta.setter
    def theta(self, value):
        self.kernel.theta = value

    def __call__(self, atoms_X, atoms_Y, dx=False, dy=False, eval_gradient=False,
            order='new'):
        n_dim = 3*len(atoms_X[0])
        X = np.zeros((len(atoms_X), n_dim))
        Y = np.zeros((len(atoms_Y), n_dim))
        for i, atoms in enumerate(atoms_X):
            X[i,:] = atoms.get_positions().flatten()
        for i, atoms in enumerate(atoms_Y):
            Y[i,:] = atoms.get_positions().flatten()
        if order=='old':
            mat = self.create_mat
        elif order=='new':
            mat = self.create_mat_new

        if not dx and not dy:
            return mat(self.kernel, X, Y, dx_max=0, dy_max=0,
                eval_gradient=eval_gradient)
        elif dx and not dy:
            return mat(self.kernel, X, Y, dx_max=n_dim, dy_max=0,
                eval_gradient=eval_gradient)
        elif not dx and dy:
            return mat(self.kernel, X, Y, dx_max=0, dy_max=n_dim,
                eval_gradient=eval_gradient)
        elif dx and dy:
            return mat(self.kernel, X, Y,
                dx_max=n_dim, dy_max=n_dim, eval_gradient=eval_gradient)

    def create_mat(self, kernel, x1, x2, dx_max=0, dy_max=0, eval_gradient=False):
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
            kernel_mat = np.zeros([n * (1 + dx_max), m * (1 + dy_max)])
            for ii in range(dx_max + 1):
                for jj in range(dy_max + 1):
                    kernel_mat[n * ii:n * (ii + 1), m * jj:m * (jj + 1)] = kernel(
                        x1, x2, dx=ii, dy=jj, eval_gradient=False)
            return kernel_mat
        else:
            num_theta = len(kernel.theta)
            kernel_derivative = np.zeros([n * (1 + dx_max), m * (1 + dy_max),
                num_theta])
            kernel_mat = np.zeros([n * (1 + dx_max), m * (1 + dy_max)])
            for ii in range(dx_max + 1):
                for jj in range(dy_max + 1):
                    k_mat, deriv_mat = kernel(x1, x2, dx=ii, dy=jj, eval_gradient=True)
                    kernel_mat[n * ii:n * (ii + 1), m * jj:m * (jj + 1)] = k_mat
                    kernel_derivative[n * ii:n * (ii + 1), m * jj:m * (jj + 1), :] = deriv_mat
        return kernel_mat, kernel_derivative

    def create_mat_new(self, kernel, x1, x2, dx_max=0, dy_max=0, eval_gradient=False):
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
            kernel_mat = np.zeros([n * (1 + dx_max), m * (1 + dy_max)])
            #kernel_mat[0:n, 0:m] = kernel(x1, x2, dx=0, dy=0, eval_gradient=False)
            for ii in range(dx_max + 1):
                for jj in range(dy_max + 1):
                    if ii == 0 and jj == 0:
                        kernel_mat[0:n, 0:m] = kernel(x1, x2, dx=0, dy=0,
                            eval_gradient=False)
                    elif ii == 0:
                        kernel_mat[0:n, (m+jj-1)::dy_max] = kernel(
                            x1, x2, dx=0, dy=jj, eval_gradient=False)
                    elif jj == 0:
                        kernel_mat[(n+ii-1)::dx_max, 0:m] = kernel(
                            x1, x2, dx=ii, dy=0, eval_gradient=False)
                    #kernel_mat[(n+ii)::dx_max, (m+jj)::dy_max] = kernel(
                    #    x1, x2, dx=ii+1, dy=jj+1, eval_gradient=False)
                    else:
                        kernel_mat[(n+ii-1)::dx_max, (m+jj-1)::dy_max] = kernel(
                            x1, x2, dx=ii, dy=jj, eval_gradient=False)
            return kernel_mat
        else:
            num_theta = len(kernel.theta)
            kernel_derivative = np.zeros([n * (1 + dx_max), m * (1 + dy_max),
                num_theta])
            kernel_mat = np.zeros([n * (1 + dx_max), m * (1 + dy_max)])
            for ii in range(dx_max + 1):
                for jj in range(dy_max + 1):
                    k_mat, deriv_mat = kernel(x1, x2, dx=ii, dy=jj, eval_gradient=True)
                    if ii==0 and jj==0:
                        kernel_mat[0:n, 0:m] = k_mat
                        kernel_derivative[0:n, 0:m] = deriv_mat
                    elif ii==0:
                        kernel_mat[0:n, (m+jj-1)::dy_max] = k_mat
                        kernel_derivative[0:n, (m+jj-1)::dy_max] = deriv_mat
                    elif jj==0:
                        kernel_mat[(n+ii-1)::dx_max, 0:m] = k_mat
                        kernel_derivative[(n+ii-1)::dx_max, 0:m] = deriv_mat
                    else:
                        kernel_mat[(n+ii-1)::dx_max, (m+jj-1)::dy_max] = k_mat
                        kernel_derivative[(n+ii-1)::dx_max, (m+jj-1)::dy_max] = deriv_mat
        return kernel_mat, kernel_derivative

class SFSKernel():
    def __init__(self, descriptor_set, factor=1.0, constant=1.0,
            kernel='dot_product'):
        self.descriptor_set = descriptor_set
        self.factor = factor
        self.constant = constant
        self.kernel = kernel

    @property
    def bounds(self):
        return (1e-5, 1e5)

    @property
    def theta(self):
        return np.array([self.factor])

    @theta.setter
    def theta(self, theta):
        self.factor = theta

    def __call__(self, atoms_X, atoms_Y, dx=False, dy=False, eval_gradient=False):
        n_dim = 3*len(atoms_X[0])
        if dx:
            Gs_X, dGs_X = zip(*[self.descriptor_set.eval_ase(
                atoms, derivatives=True) for atoms in atoms_X])
        else:
            Gs_X = [self.descriptor_set.eval_ase(atoms) for atoms in atoms_X]
        if dy:
            Gs_Y, dGs_Y = zip(*[self.descriptor_set.eval_ase(
                atoms, derivatives=True) for atoms in atoms_Y])
        else:
            Gs_Y = [self.descriptor_set.eval_ase(atoms) for atoms in atoms_Y]
        types_X = [atoms.get_chemical_symbols() for atoms in atoms_X]
        types_Y = [atoms.get_chemical_symbols() for atoms in atoms_Y]

        descrip_dim = len(Gs_X[0][0])
        n = len(atoms_X)
        m = len(atoms_Y)
        if not eval_gradient:
            if dx and dy:
                kernel_mat = np.zeros((n*(1+n_dim), m*(1+n_dim)))
                kernel_mat[:n,:m] += self.constant
                for i, (Gsi, dGsi, tsi) in enumerate(zip(Gs_X, dGs_X, types_X)):
                    for j, (Gsj, dGsj, tsj) in enumerate(zip(Gs_Y, dGs_Y, types_Y)):
                        for Gi, dGi, ti in zip(Gsi, dGsi, tsi):
                            norm_Gi = np.linalg.norm(Gi)
                            for Gj, dGj, tj in zip(Gsj, dGsj, tsj):
                                if ti == tj:
                                    norm_Gj = np.linalg.norm(Gj)
                                    if self.kernel == 'dot_product':
                                        kernel_mat[i,j] += Gi.dot(Gj)
                                        K = Gj
                                        K_prime = Gi
                                        J = np.eye(descrip_dim)
                                    elif self.kernel == 'dot_product_norm':
                                        kernel_mat[i,j] += Gi.dot(Gj)/(norm_Gi*norm_Gj)
                                        K = (-Gi.dot(Gj)*Gi+norm_Gi**2*Gj)/(norm_Gi**3*norm_Gj)
                                        K_prime = (-Gi.dot(Gj)*Gj+norm_Gj**2*Gi)/(norm_Gj**3*norm_Gi)
                                        J = (-np.outer(Gj,Gj)*norm_Gi**2 - np.outer(Gi,Gi)*norm_Gj**2+
                                            np.outer(Gj,Gi)*np.dot(Gi,Gj)+np.eye(descrip_dim)*norm_Gi**2*norm_Gj**2
                                            )/(norm_Gi**3*norm_Gj**3)
                                    elif self.kernel == 'squared_exp':
                                        exp_mat = np.exp(-np.sum((Gi-Gj)**2)/2)
                                        kernel_mat[i,j] += exp_mat
                                        K = -exp_mat*(Gi-Gj)
                                        K_prime = -exp_mat*(Gj-Gi)
                                        J = exp_mat*(np.eye(descrip_dim) - np.outer(Gi,Gi) + np.outer(Gi,Gj) +
                                            np.outer(Gj,Gi) - np.outer(Gj,Gj))
                                    else:
                                        raise NotImplementedError
                                    kernel_mat[n+i*n_dim:n+(i+1)*n_dim,j] += K.dot(dGi.reshape((-1,n_dim)))
                                    kernel_mat[i, m+j*n_dim:m+(j+1)*n_dim] += K_prime.dot(dGj.reshape((-1,n_dim)))
                                    kernel_mat[n+i*n_dim:n+(i+1)*n_dim,m+j*n_dim:m+(j+1)*n_dim] += (
        			                    dGi.reshape((-1,n_dim)).T.dot(J).dot(dGj.reshape((-1,n_dim))))
            else:
                raise NotImplementedError
            return kernel_mat
        else:
            raise NotImplementedError
