from sklearn.gaussian_process.kernels import RBF as sk_RBF
from sklearn.gaussian_process.kernels import Sum as sk_Sum
from sklearn.gaussian_process.kernels import ConstantKernel as sk_ConstantKernel
from sklearn.gaussian_process.kernels import _check_length_scale
from scipy.spatial.distance import pdist, cdist, squareform
import numpy as np

class Sum(sk_Sum):
    def __call__(self, X, Y=None, dx=0, dy=0, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if eval_gradient:
            K1, K1_gradient = self.k1(X, Y, dx=dx, dy=dy, eval_gradient=True)
            K2, K2_gradient = self.k2(X, Y, dx=dx, dy=dy, eval_gradient=True)
            return K1 + K2, np.dstack((K1_gradient, K2_gradient))
        else:
            return self.k1(X, Y, dx=dx, dy=dy) + self.k2(X, Y, dx=dx, dy=dy)

class ConstantKernel(sk_ConstantKernel):
    def __call__(self, X, Y=None, dx=0, dy=0, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        elif eval_gradient:
            if not X is Y:
                raise ValueError("Gradient can only be evaluated when Y is None or X is Y.")

        if dx==0 and dy==0:
            K = np.full((X.shape[0], Y.shape[0]), self.constant_value,
                        dtype=np.array(self.constant_value).dtype)
        else:
            K = np.zeros((X.shape[0], Y.shape[0]),
                        dtype=np.array(self.constant_value).dtype)
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed and dx==0 and dy==0:
                return (K, np.full((X.shape[0], X.shape[0], 1),
                                   self.constant_value,
                                   dtype=np.array(self.constant_value).dtype))
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K


class RBF(sk_RBF):
    def __call__(self, X, Y=None, dx=0, dy=0, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        #X = atoms_X.get_positions()
        #if atoms_Y is None:
        #    Y = X
        #else:
        #    Y = atoms_Y.get_positions()
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        dists = cdist(X / length_scale, Y / length_scale, metric='sqeuclidean')
        K = np.exp(-.5 * dists)

        if not self.anisotropic or length_scale.shape[0] == 1:
            if dx != 0 or dy != 0:
                if dx == dy:
                    K = K * (1 - np.subtract.outer(X[:, dx-1].T, Y[:, dx-1])**2 / length_scale**2) / length_scale**2 # J_ii
                elif dx == 0: # and dy != 0:
                    K = K * np.subtract.outer(X[:, dy-1].T, Y[:, dy-1])/length_scale**2 # G
                elif dy == 0: # and dx != 0:
                    K = -K * np.subtract.outer(X[:, dx-1].T, Y[:, dx-1])/length_scale**2 # K_prime
                else:
                    K = -K * np.subtract.outer(X[:, dx-1].T, Y[:, dx-1])*np.subtract.outer(X[:, dy-1].T, Y[:, dy-1])\
                                 /length_scale**4 # J_ij
        else:
            if dx != 0 or dy != 0:
                if dx == dy:
                    K = K * (1 - np.subtract.outer(X[:,dx-1].T, Y[:, dx-1])**2/length_scale[dx-1]**2)\
                             /length_scale[dx-1]**2
                elif dx == 0:
                    K = K * np.subtract.outer(X[:, dy-1].T, Y[:, dy-1])/length_scale[dy-1]**2
                elif dy == 0:
                    K = -K * np.subtract.outer(X[:, dx-1].T, Y[:, dx-1])/length_scale[dx-1]**2
                else:
                    K = - K * np.subtract.outer(X[:, dx-1].T, Y[:, dx-1])*np.subtract.outer(X[:, dy-1].T, Y[:, dy-1])\
                             / (length_scale[dx-1]**2 * length_scale[dy-1]**2)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], Y.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                if dx == 0 and dy == 0:
                    K_gradient = (K * dists)[:, :, np.newaxis]
                elif dx != 0 and dy == 0:
                    K_gradient = (K * (dists - 2))[:, :, np.newaxis]
                elif dy != 0 and dx == 0:
                    K_gradient = (K * (dists - 2))[:, :, np.newaxis]
                    # K_gradient = (K * (dists/length_scale**2 - 2/length_scale**2))[:, :, np.newaxis]
                else:
                    if dx == dy:
                        K_gradient = (K * (dists - 4) + 2 * np.exp(-0.5*dists)/length_scale**2)[:, :, np.newaxis]
                        # K_gradient = (np.exp(-.5 * dists) / length_scale ** 2 * (5*dists - 2 - dists **2))[:, :, np.newaxis]
                        # K_gradient = (np.exp(-0.5*dists)*(dists**2/length_scale**3-1./length_scale**5-4*(dists-1/length_scale**3)-2/length_scale**3))[:, :, np.newaxis]
                    else:
                        K_gradient = (K * (dists - 4))[:, :, np.newaxis]
                        # K_gradient = (-np.exp(-.5 * dists) * dists/length_scale**2 * (4 - dists))[:, :, np.newaxis]
                        # K_gradient = (np.exp(-.5 * dists) * dists**2 / length_scale ** 3 * (dists**2 - 4))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:

                # We need to recompute the pairwise dimension-wise distances
                grad = (X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2 / (length_scale ** 2)
                if dx == 0 and dy == 0:
                    K_gradient = grad * K[..., np.newaxis]

                elif (dx != 0 and dy == 0):
                    K_gradient = grad * K[..., np.newaxis]
                    K_gradient[:, :, dx - 1] = K_gradient[:, :, dx - 1] - 2 * K
                elif (dy != 0 and dx == 0):
                    K_gradient = grad * K[..., np.newaxis]
                    K_gradient[:, :, dy-1] = K_gradient[:, :, dy-1] - 2*K
                else:
                    if dx == dy:
                        K_gradient = grad * K[..., np.newaxis]
                        K_gradient[:, :, dx - 1] = K_gradient[:, :, dx - 1] - np.exp(-0.5 * dists) * (2 /
                                            length_scale[dx-1] ** 2 - 4 * grad[:, :, dx - 1] / length_scale[dx-1] ** 2)

                    else:
                        K_gradient = grad * K[..., np.newaxis]
                        K_gradient[:, :, dx - 1] = K_gradient[:, :, dx - 1] - 2*K
                        K_gradient[:, :, dy - 1] = K_gradient[:, :, dy - 1] - 2*K
                return K, K_gradient
        else:
            return K
