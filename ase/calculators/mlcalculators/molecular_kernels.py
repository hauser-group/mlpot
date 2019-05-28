import numpy as np
try:
    from numba import jit
except ImportError:
    import warnings
    warnings.warn("Could not import numba!")
    # Dummy jit decorator
    def jit(nopython=True):
        def jit_decorator(func):
            def wrapper():
                return func()
            return wrapper
        return jit_decorator

class DotProductKernel():

    def __init__(self, constant=1.0, exponent=2):
        self.constant = constant
        self.exponent = exponent

    @property
    def theta(self):
        return np.empty(0)

    @theta.setter
    def theta(self, theta):
        pass

    @property
    def bounds(self):
        return np.empty((0,2))

    def __call__(self, X, Y, dx=False, dy=False, eval_gradient=False):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]

        # The arguments dx and dy are deprecated and will be removed soon
        if not (dx and dy):
            raise NotImplementedError
        # Initialize kernel matrix
        K = np.zeros((n*(1+n_dim), m*(1+n_dim)))
        if eval_gradient:
            K_gradient = np.zeros((n*(1+n_dim), m*(1+n_dim), 1))
        for a in range(n):
            for b in range(m):
                # Index ranges for the derivatives are given by the following
                # slice objects:
                da = slice(n+a*n_dim, n+(a+1)*n_dim, 1)
                db = slice(m+b*n_dim, m+(b+1)*n_dim, 1)

                dot_plus_c = X[a,:].dot(Y[b,:]) + self.constant

                K[a, b] = dot_plus_c**self.exponent
                K[da, b] = dot_plus_c**(self.exponent - 1)*self.exponent*Y[b,:]
                K[a, db] = dot_plus_c**(self.exponent - 1)*self.exponent*X[a,:]
                K[da, db] = self.exponent*dot_plus_c**(self.exponent - 2)*(
                    (self.exponent - 1)*np.outer(Y[b,:],X[a,:]) +
                    np.eye(n_dim)*dot_plus_c)

        if not eval_gradient:
            return K
        else:
            return K, K_gradient

class NormalizedDotProductKernel():

    def __init__(self, constant=1.0, exponent=2, additiv_constant=0.0):
        self.constant = constant
        self.exponent = exponent
        self.additiv_constant = additiv_constant

    @property
    def theta(self):
        return np.empty(0)

    @theta.setter
    def theta(self, theta):
        pass

    @property
    def bounds(self):
        return np.empty((0,2))

    def __call__(self, X, Y, dx=False, dy=False, eval_gradient=False):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]

        # The arguments dx and dy are deprecated and will be removed soon
        if not (dx and dy):
            raise NotImplementedError
        # Initialize kernel matrix
        K = np.zeros((n*(1+n_dim), m*(1+n_dim)))
        if eval_gradient:
            K_gradient = np.zeros((n*(1+n_dim), m*(1+n_dim), 1))
        for a in range(n):
            for b in range(m):
                # Index ranges for the derivatives are given by the following
                # slice objects:
                da = slice(n+a*n_dim, n+(a+1)*n_dim, 1)
                db = slice(m+b*n_dim, m+(b+1)*n_dim, 1)

                dot_plus_c = X[a,:].dot(Y[b,:]) + self.constant
                xx_plus_c = X[a,:].dot(X[a,:]) + self.constant
                yy_plus_c = Y[b,:].dot(Y[b,:]) + self.constant
                norm_xx = np.sqrt(xx_plus_c**self.exponent)
                norm_yy = np.sqrt(yy_plus_c**self.exponent)

                K[a, b] = dot_plus_c**self.exponent/(norm_xx*norm_yy)
                K[da, b] = self.exponent*dot_plus_c**(self.exponent - 1)*(
                    xx_plus_c*Y[b,:] - dot_plus_c*X[a,:])/(
                    xx_plus_c*norm_xx*norm_yy)
                K[a, db] = self.exponent*dot_plus_c**(self.exponent - 1)*(
                    yy_plus_c*X[a,:] - dot_plus_c*Y[b,:])/(
                    yy_plus_c*norm_xx*norm_yy)
                K[da, db] = self.exponent*dot_plus_c**(self.exponent - 1)*(
                    np.eye(n_dim) +
                    (self.exponent - 1)*np.outer(Y[b,:], X[a,:])/(dot_plus_c) +
                    self.exponent*np.outer(X[a,:], Y[b,:])*dot_plus_c/(
                    xx_plus_c*yy_plus_c) -
                    self.exponent*np.outer(X[a,:], X[a,:])/xx_plus_c -
                    self.exponent*np.outer(Y[b,:], Y[b,:])/yy_plus_c)/(
                    norm_xx*norm_yy)

        # Add constant term only on non-derivative block
        K[:n,:m] += self.additiv_constant

        if not eval_gradient:
            return K
        else:
            return K, K_gradient


class RBFKernel():

    def __init__(self, constant=0.0, factor=1.0, length_scale=np.array([1.0]),
            length_scale_bounds=(1e-3, 1e3)):
        self.factor = factor
        self.constant = constant
        if np.ndim(length_scale) == 0:
            self.length_scale = np.array([length_scale])
        elif np.ndim(length_scale) == 1:
            self.length_scale = np.array(length_scale)
        else:
            raise ValueError('Unexpected dimension of length_scale')
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def theta(self):
        return np.log(self.length_scale)

    @theta.setter
    def theta(self, theta):
        self.length_scale = np.exp(theta)

    @property
    def bounds(self):
        if np.ndim(self.length_scale_bounds) == 1:
            return np.log(np.asarray([self.length_scale_bounds]))
        elif np.ndim(self.length_scale_bounds) == 2:
            return np.log(self.length_scale_bounds)
        else:
            raise ValueError('Unexpected dimension of length_scale_bounds')

    def __call__(self, X, Y, dx=False, dy=False, eval_gradient=False, method='old'):
        if method == 'old':
            return self._eval_old(X, Y, self.anisotropic, self.length_scale, self.factor,
                self.constant, dx=dx, dy=dy, eval_gradient=eval_gradient)
        elif method == 'new':
            if eval_gradient:
                return self._eval_with_gradient(X, Y, self.anisotropic,
                    self.length_scale, self.factor, self.constant, dx=dx, dy=dy)
            else:
                return self._eval(X, Y, self.anisotropic, self.length_scale,
                    self.factor, self.constant, dx=dx, dy=dy)

    @staticmethod
    @jit(nopython=True)
    def _eval(X, Y, anisotropic, length_scale, factor, constant,
        dx=False, dy=False, eval_gradient=False):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]

        # The arguments dx and dy are deprecated and will be removed soon
        if not (dx and dy):
            raise NotImplementedError
        # Initialize kernel matrix
        K = np.zeros((n*(1+n_dim), m*(1+n_dim)))
        for a in range(n):
            for b in range(m):
                # Index ranges for the derivatives are given by the following
                # slice objects:
                da = slice(n+a*n_dim, n+(a+1)*n_dim, 1)
                db = slice(m+b*n_dim, m+(b+1)*n_dim, 1)
                # A few helpful quantities:
                scaled_diff = (X[a,:]-Y[b,:])/length_scale
                inner_prod = scaled_diff.dot(scaled_diff)
                outer_prod_over_l = np.outer(scaled_diff/length_scale,
                    scaled_diff/length_scale)
                exp_term = np.exp(-.5*inner_prod)
                delta_qp = np.eye(n_dim)
                # populate kernel matrix:
                K[a, b] = exp_term
                K[da, b] = -exp_term*scaled_diff/length_scale
                K[a, db] = exp_term*scaled_diff/length_scale
                K[da, db] = exp_term*(
                    delta_qp/length_scale**2 - outer_prod_over_l)

        # Multiply by factor
        K *= factor
        # Add constant term only on non-derivative block
        K[:n,:m] += constant
        return K

    @staticmethod
    @jit(nopython=True)
    def _eval_with_gradient(X, Y, anisotropic, length_scale, factor, constant,
        dx=False, dy=False):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]

        # The arguments dx and dy are deprecated and will be removed soon
        if not (dx and dy):
            raise NotImplementedError
        # Initialize kernel matrix
        K = np.zeros((n*(1+n_dim), m*(1+n_dim)))

        # Array to hold the derivatives with respect to the length_scale
        if anisotropic:
            K_gradient = np.zeros((n*(1+n_dim), m*(1+n_dim),
                length_scale.shape[0]))
        else: # isotropic
            K_gradient = np.zeros((n*(1+n_dim), m*(1+n_dim), 1))
        for a in range(n):
            for b in range(m):
                # Index ranges for the derivatives are given by the following
                # slice objects:
                da = slice(n+a*n_dim, n+(a+1)*n_dim, 1)
                db = slice(m+b*n_dim, m+(b+1)*n_dim, 1)
                # A few helpful quantities:
                scaled_diff = (X[a,:]-Y[b,:])/length_scale
                inner_prod = scaled_diff.dot(scaled_diff)
                outer_prod = np.outer(scaled_diff, scaled_diff)
                outer_prod_over_l = np.outer(scaled_diff/length_scale,
                    scaled_diff/length_scale)
                exp_term = np.exp(-.5*inner_prod)
                delta_qp = np.eye(n_dim)
                # populate kernel matrix:
                K[a, b] = exp_term
                K[da, b] = -exp_term*scaled_diff/length_scale
                K[a, db] = exp_term*scaled_diff/length_scale
                K[da, db] = exp_term*(
                    delta_qp/length_scale**2 - outer_prod_over_l)

                if anisotropic:
                    # Following the accompaning latex documents the
                    # three matrix dimensions are refered to as q, p and s.
                    K_gradient[a, b, :] = exp_term*(
                        scaled_diff**2/length_scale)
                    K_gradient[da, b, :] = exp_term*(
                        2*np.diag(scaled_diff/length_scale**2) -
                        np.outer(scaled_diff/length_scale,
                        scaled_diff**2/length_scale))
                    K_gradient[a, db, :] = -exp_term*(
                        2*np.diag(scaled_diff/length_scale**2) -
                        np.outer(scaled_diff/length_scale,
                        scaled_diff**2/length_scale))
                    for s in range(n_dim):
                        delta_qs = np.zeros((n_dim, n_dim))
                        delta_qs[s,:] = 1.0
                        delta_ps = np.zeros((n_dim, n_dim))
                        delta_ps[:,s] = 1.0
                        K_gradient[da, db, s] = exp_term*(
                            delta_qp*(scaled_diff[s]**2 - 2*delta_qs) +
                            outer_prod*(2*delta_qs+2*delta_ps - scaled_diff[s]**2)
                        )/(length_scale[s]*np.outer(length_scale, length_scale))
                else: # isotropic
                    K_gradient[a, b, 0] = exp_term*(
                        inner_prod/length_scale[0])
                    K_gradient[da, b, 0] = exp_term*(
                        scaled_diff*(2 - inner_prod)/length_scale**2)
                    K_gradient[a, db, 0] = -exp_term*(
                        scaled_diff*(2 - inner_prod)
                        /length_scale**2)
                    K_gradient[da, db, 0] = exp_term*(
                        delta_qp*(inner_prod - 2) +
                        outer_prod*(4 - inner_prod))/length_scale**3


        # Multiply gradient with respect to the length_scale by factor
        K_gradient *= factor

        # Multiply by factor
        K *= factor
        # Add constant term only on non-derivative block
        K[:n,:m] += constant

        return K, K_gradient

    @staticmethod
    def _eval_old(X, Y, anisotropic, length_scale, factor, constant,
        dx=False, dy=False, eval_gradient=False):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]

        # The arguments dx and dy are deprecated and will be removed soon
        if not (dx and dy):
            raise NotImplementedError
        # Initialize kernel matrix
        K = np.zeros((n*(1+n_dim), m*(1+n_dim)))
        if eval_gradient:
            # Array to hold the derivatives with respect to the length_scale
            if anisotropic:
                K_gradient = np.zeros((n*(1+n_dim), m*(1+n_dim),
                    length_scale.shape[0]))
            else: # isotropic
                K_gradient = np.zeros((n*(1+n_dim), m*(1+n_dim), 1))
        for a in range(n):
            for b in range(m):
                # Index ranges for the derivatives are given by the following
                # slice objects:
                da = slice(n+a*n_dim, n+(a+1)*n_dim, 1)
                db = slice(m+b*n_dim, m+(b+1)*n_dim, 1)
                # A few helpful quantities:
                scaled_diff = ((X[a,:]-Y[b,:])/length_scale)
                inner_prod = scaled_diff.dot(scaled_diff)
                outer_prod_over_l = np.outer(scaled_diff/length_scale,
                    scaled_diff/length_scale)
                exp_term = np.exp(-.5*inner_prod)
                # populate kernel matrix:
                K[a, b] = exp_term
                K[da, b] = -exp_term*scaled_diff/length_scale
                K[a, db] = exp_term*scaled_diff/length_scale
                K[da, db] = exp_term*(
                    np.eye(n_dim)/length_scale**2 - outer_prod_over_l)

                # Gradient with respect to the length_scale
                if eval_gradient:
                    if anisotropic:
                        # Following the accompaning latex documents the
                        # three matrix dimensions are refered to as q, p and s.
                        K_gradient[a, b, :] = exp_term*(
                            scaled_diff**2/length_scale)
                        K_gradient[da, b, :] = exp_term*(
                            2*np.diag(scaled_diff/length_scale**2) -
                            np.outer(scaled_diff/length_scale,
                            scaled_diff**2/length_scale))
                        K_gradient[a, db, :] = -exp_term*(
                            2*np.diag(scaled_diff/length_scale**2) -
                            np.outer(scaled_diff/length_scale,
                            scaled_diff**2/length_scale))
                        delta_qp_over_lq2 = np.repeat((np.eye(n_dim)/
                            length_scale**2)[:, :, np.newaxis],
                            n_dim, axis=2)
                        delta_qs = np.repeat(
                            np.eye(n_dim)[:, np.newaxis, :], n_dim, axis=1)
                        delta_ps = np.repeat(
                            np.eye(n_dim)[np.newaxis, :, :], n_dim, axis=0)
                        scaled_diff_s_squared = np.tile(
                            scaled_diff**2, (n_dim, n_dim, 1))
                        K_gradient[da, db, :] = exp_term*(delta_qp_over_lq2*(
                            scaled_diff_s_squared - 2*delta_qs) +
                            np.repeat(np.outer(scaled_diff/length_scale,
                            scaled_diff/length_scale)[:, :, np.newaxis],
                            n_dim, axis=2)*
                            (2*delta_qs + 2*delta_ps - scaled_diff_s_squared)
                            )/length_scale
                    else: # isotropic
                        outer_prod = np.outer(scaled_diff, scaled_diff)
                        K_gradient[a, b, 0] = exp_term*(
                            inner_prod/length_scale)
                        K_gradient[da, b, 0] = exp_term*(
                            scaled_diff*(2 - inner_prod)/length_scale**2)
                        K_gradient[a, db, 0] = -exp_term*(
                            scaled_diff*(2 - inner_prod)
                            /length_scale**2)
                        K_gradient[da, db, 0] = exp_term*(
                            np.eye(n_dim)*(inner_prod - 2) +
                            outer_prod*(4 - inner_prod))/length_scale**3

        if eval_gradient:
            # Multiply gradient with respect to the length_scale by factor
            K_gradient *= factor

        # Multiply by factor
        K *= factor
        # Add constant term only on non-derivative block
        K[:n,:m] += constant

        if not eval_gradient:
            return K
        else:
            return K, K_gradient

class RBFKernel_with_factor():

    def __init__(self, constant=0.0, factor=1.0, length_scale=1.0,
            factor_bounds=(1e-5, 1e5), length_scale_bounds=(1e-3, 1e3)):
        self.factor = factor
        self.constant = constant
        self.length_scale = length_scale
        self.factor_bounds = factor_bounds
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def theta(self):
        return np.log(np.hstack([self.factor, self.length_scale]))

    @theta.setter
    def theta(self, theta):
        self.factor = np.exp(theta[0])
        self.length_scale = np.exp(theta[1:])

    @property
    def bounds(self):
        if np.ndim(self.length_scale_bounds) == 1:
            return np.log(np.asarray(
                [self.factor_bounds, self.length_scale_bounds]))
        elif np.ndim(self.length_scale_bounds) == 2:
            return np.log(np.asarray(
                [self.factor_bounds]+self.length_scale_bounds))

    def __call__(self, X, Y, dx=False, dy=False, eval_gradient=False):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]

        # The arguments dx and dy are deprecated and will be removed soon
        if not (dx and dy):
            raise NotImplementedError
        # Initialize kernel matrix
        K = np.zeros((n*(1+n_dim), m*(1+n_dim)))
        if eval_gradient:
            # Array to hold the derivatives with respect to the factor and
            # the length_scale
            if self.anisotropic:
                K_gradient = np.zeros((n*(1+n_dim), m*(1+n_dim), 1 +
                    self.length_scale.shape[0]))
            else: # isotropic
                K_gradient = np.zeros((n*(1+n_dim), m*(1+n_dim), 2))
        for a in range(n):
            for b in range(m):
                # Index ranges for the derivatives are given by the following
                # slice objects:
                da = slice(n+a*n_dim, n+(a+1)*n_dim, 1)
                db = slice(m+b*n_dim, m+(b+1)*n_dim, 1)
                # A few helpful quantities:
                scaled_diff = (X[a,:]-Y[b,:])/self.length_scale
                inner_prod = scaled_diff.dot(scaled_diff)
                outer_prod_over_l = np.outer(scaled_diff/self.length_scale,
                    scaled_diff/self.length_scale)
                exp_term = np.exp(-.5*inner_prod)
                # populate kernel matrix:
                K[a, b] = exp_term
                K[da, b] = -exp_term*scaled_diff/self.length_scale
                K[a, db] = exp_term*scaled_diff/self.length_scale
                K[da, db] = exp_term*(
                    np.eye(n_dim)/self.length_scale**2 - outer_prod_over_l)

                # Gradient with respect to the length_scale
                if eval_gradient:
                    if self.anisotropic:
                        # Following the accompaning latex documents the
                        # three matrix dimensions are refered to as q, p and s.
                        K_gradient[a, b, 1:] = exp_term*(
                            scaled_diff**2/self.length_scale)
                        K_gradient[da, b, 1:] = exp_term*(
                            2*np.diag(scaled_diff/self.length_scale**2) -
                            np.outer(scaled_diff/self.length_scale,
                            scaled_diff**2/self.length_scale))
                        K_gradient[a, db, 1:] = -exp_term*(
                            2*np.diag(scaled_diff/self.length_scale**2) -
                            np.outer(scaled_diff/self.length_scale,
                            scaled_diff**2/self.length_scale))
                        delta_qp_over_lq2 = np.repeat((np.eye(n_dim)/
                            self.length_scale**2)[:, :, np.newaxis],
                            n_dim, axis=2)
                        delta_qs = np.repeat(
                            np.eye(n_dim)[:, np.newaxis, :], n_dim, axis=1)
                        delta_ps = np.repeat(
                            np.eye(n_dim)[np.newaxis, :, :], n_dim, axis=0)
                        scaled_diff_s_squared = np.tile(
                            scaled_diff**2, (n_dim, n_dim, 1))
                        K_gradient[da, db, 1:] = exp_term*(delta_qp_over_lq2*(
                            scaled_diff_s_squared - 2*delta_qs) +
                            np.repeat(np.outer(scaled_diff/self.length_scale,
                            scaled_diff/self.length_scale)[:, :, np.newaxis],
                            n_dim, axis=2)*
                            (2*delta_qs + 2*delta_ps - scaled_diff_s_squared)
                            )/self.length_scale

                    else: # isotropic
                        outer_prod = np.outer(scaled_diff, scaled_diff)
                        K_gradient[a, b, 1] = exp_term*(
                            inner_prod/self.length_scale)
                        K_gradient[da, b, 1] = exp_term*(
                            scaled_diff*(2 - inner_prod)/self.length_scale**2)
                        K_gradient[a, db, 1] = -exp_term*(
                            scaled_diff*(2 - inner_prod)
                            /self.length_scale**2)
                        K_gradient[da, db, 1] = exp_term*(
                            np.eye(n_dim)*(inner_prod - 2) +
                            outer_prod*(4 - inner_prod))/self.length_scale**3

        if eval_gradient:
            # Gradient with respect to the factor
            K_gradient[:,:,0] = K
            # Multiply gradient with respect to the length_scale by factor
            K_gradient[:,:,1:] *= self.factor

        # Multiply by factor
        K *= self.factor
        # Add constant term only on non-derivative block
        K[:n,:m] += self.constant

        if not eval_gradient:
            return K
        else:
            return K, K_gradient

class MaternKernel():
    def __init__(self, constant=0.0, factor=1.0, length_scale=np.array([1.0]),
            length_scale_bounds=(1e-3, 1e3)):
        self.factor = factor
        self.constant = constant
        if np.ndim(length_scale) == 0:
            self.length_scale = np.array([length_scale])
        elif np.ndim(length_scale) == 1:
            self.length_scale = np.array([length_scale])
        else:
            raise ValueError('Unexpected dimension of length_scale')
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def theta(self):
        return np.log(self.length_scale)

    @theta.setter
    def theta(self, theta):
        self.length_scale = np.exp(theta)

    @property
    def bounds(self):
        if np.ndim(self.length_scale_bounds) == 1:
            return np.log(np.asarray([self.length_scale_bounds]))
        elif np.ndim(self.length_scale_bounds) == 2:
            return np.log(self.length_scale_bounds)
        else:
            raise ValueError('Unexpected dimension of length_scale_bounds')

    def __call__(self, X, Y, dx=False, dy=False, eval_gradient=False):
        n = X.shape[0]
        m = Y.shape[0]
        n_dim = X.shape[1]

        # The arguments dx and dy are deprecated and will be removed soon
        if not (dx and dy):
            raise NotImplementedError
        # Initialize kernel matrix
        K = np.zeros((n*(1+n_dim), m*(1+n_dim)))
        if eval_gradient:
            # Array to hold the derivatives with respect to the length_scale
            if self.anisotropic:
                K_gradient = np.zeros((n*(1+n_dim), m*(1+n_dim),
                    self.length_scale.shape[0]))
            else: # isotropic
                K_gradient = np.zeros((n*(1+n_dim), m*(1+n_dim), 1))
        for a in range(n):
            for b in range(m):
                # Index ranges for the derivatives are given by the following
                # slice objects:
                da = slice(n+a*n_dim, n+(a+1)*n_dim, 1)
                db = slice(m+b*n_dim, m+(b+1)*n_dim, 1)
                # A few helpful quantities:
                scaled_diff = (X[a,:]-Y[b,:])/self.length_scale
                inner_prod = scaled_diff.dot(scaled_diff)
                sqrt_sum = np.sqrt(5.*inner_prod)
                outer_prod_over_l = np.outer(scaled_diff/self.length_scale,
                    scaled_diff/self.length_scale)
                exp_term = np.exp(-sqrt_sum)
                # populate kernel matrix:
                K[a, b] = (1. + sqrt_sum + 5./3.*inner_prod)*exp_term
                K[da, b] = -5./3.*(
                    1. + sqrt_sum)*scaled_diff/self.length_scale*exp_term
                K[a, db] = 5./3.*(
                    1. + sqrt_sum)*scaled_diff/self.length_scale*exp_term
                K[da, db] = 5./3.*(np.eye(n_dim)/self.length_scale**2*(
                    sqrt_sum + 1.0) - 5*outer_prod_over_l)*exp_term

                # Gradient with respect to the length_scale
                if eval_gradient:
                    if self.anisotropic:
                        raise NotImplementedError
                    else: # isotropic
                        outer_prod = np.outer(scaled_diff, scaled_diff)
                        K_gradient[a, b, 0] = 5./3.*(sqrt_sum + 1)*(
                            inner_prod/self.length_scale)*exp_term
                        K_gradient[da, b, 0] = 5./3.*(2. - 5.*inner_prod + 2.*
                            sqrt_sum)*scaled_diff/self.length_scale**2*exp_term
                        K_gradient[a, db, 0] = -5./3.*(2. - 5.*inner_prod + 2.*
                            sqrt_sum)*scaled_diff/self.length_scale**2*exp_term
                        K_gradient[da, db, 0] = 5./3.*(np.eye(n_dim)*(
                            5.*inner_prod - 2.*sqrt_sum - 2.) + 5.*outer_prod*(
                            4. - sqrt_sum))/self.length_scale**3*exp_term

        if eval_gradient:
            # Multiply gradient with respect to the length_scale by factor
            K_gradient *= self.factor

        # Multiply by factor
        K *= self.factor
        # Add constant term only on non-derivative block
        K[:n,:m] += self.constant

        if not eval_gradient:
            return K
        else:
            return K, K_gradient

class SFSKernel():
    def __init__(self, descriptor_set, factor=1.0, constant=1.0,
            kernel='dot_product'):
        self.descriptor_set = descriptor_set
        self.factor = factor
        self.constant = constant
        self.kernel = kernel

    @property
    def bounds(self):
        return (1e-3, 1e3)

    @property
    def theta(self):
        return np.log(np.array([self.factor]))

    @theta.setter
    def theta(self, theta):
        self.factor = np.exp(theta[0])

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
                for i, (Gsi, dGsi, tsi) in enumerate(zip(Gs_X, dGs_X, types_X)):
                    for j, (Gsj, dGsj, tsj) in enumerate(zip(Gs_Y, dGs_Y, types_Y)):
                        for Gi, dGi, ti in zip(Gsi, dGsi, tsi):
                            norm_Gi = np.linalg.norm(Gi)
                            for Gj, dGj, tj in zip(Gsj, dGsj, tsj):
                                if ti == tj:
                                    norm_Gj = np.linalg.norm(Gj)
                                    kernel_mat[i,j] += self.constant
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
                                        K_prime = exp_mat*(Gi-Gj)
                                        J = exp_mat*(np.eye(descrip_dim) - np.outer(Gi,Gi) + np.outer(Gi,Gj) +
                                            np.outer(Gj,Gi) - np.outer(Gj,Gj))
                                    else:
                                        raise NotImplementedError
                                    kernel_mat[n+i*n_dim:n+(i+1)*n_dim,j] += K.dot(dGi.reshape((-1,n_dim)))
                                    kernel_mat[i, m+j*n_dim:m+(j+1)*n_dim] += K_prime.dot(dGj.reshape((-1,n_dim)))
                                    kernel_mat[n+i*n_dim:n+(i+1)*n_dim,m+j*n_dim:m+(j+1)*n_dim] += (
        			                    dGi.reshape((-1,n_dim)).T.dot(J).dot(dGj.reshape((-1,n_dim))))
                return kernel_mat
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

class SFSKernel_new():
    def __init__(self, descriptor_set, factor=1.0, constant=1.0,
            kernel='dot_product'):
        self.descriptor_set = descriptor_set
        self.factor = factor
        self.constant = constant
        self.kernel = kernel
        self.dim_descriptor = sum(self.descriptor_set.num_Gs)
        self.descriptor_pos = {}
        prev = 0
        for ti in self.descriptor_set.atomtypes:
            int_ti = self.descriptor_set.type_dict[ti]
            self.descriptor_pos[ti] = (
                prev, prev + self.descriptor_set.num_Gs[int_ti])
            prev = prev + self.descriptor_set.num_Gs[int_ti]

    @property
    def bounds(self):
        return (1e-3, 1e3)

    @property
    def theta(self):
        return np.log(np.array([self.factor]))

    @theta.setter
    def theta(self, theta):
        self.factor = np.exp(theta[0])

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

        print(len(Gs_X),len(Gs_Y))
        n = len(atoms_X)
        m = len(atoms_Y)
        if not eval_gradient:
            if dx and dy:
                kernel_mat = np.zeros((n*(1+n_dim), m*(1+n_dim)))
                for i, (Gsi, dGsi, tsi) in enumerate(zip(Gs_X, dGs_X, types_X)):
                    for j, (Gsj, dGsj, tsj) in enumerate(zip(Gs_Y, dGs_Y, types_Y)):
                        # TODO: this calculation should take place at the beginning
                        # of the method...
                        num_atoms_i = np.zeros(len(self.descriptor_set.atomtypes))
                        di = np.zeros(self.dim_descriptor)
                        ddi = np.zeros((3*len(Gsi), self.dim_descriptor))
                        for Gi, dGi, ti in zip(Gsi, dGsi, tsi):
                            num_atoms_i[self.descriptor_set.type_dict[ti]] += self.constant
                            di[self.descriptor_pos[ti][0]:self.descriptor_pos[ti][1]] += Gi
                            ddi[:, self.descriptor_pos[ti][0]:self.descriptor_pos[ti][1]] += dGi.reshape((-1,3*len(Gsi))).T

                        num_atoms_j = np.zeros(len(self.descriptor_set.atomtypes))
                        dj = np.zeros(self.dim_descriptor)
                        ddj = np.zeros((3*len(Gsj), self.dim_descriptor))
                        for Gj, dGj, tj in zip(Gsj, dGsj, tsj):
                            num_atoms_j[self.descriptor_set.type_dict[tj]] += self.constant
                            dj[self.descriptor_pos[tj][0]:self.descriptor_pos[tj][1]] += Gj
                            ddj[:, self.descriptor_pos[tj][0]:self.descriptor_pos[tj][1]] += dGj.reshape((-1,3*len(Gsj))).T
                        kernel_mat[i,j] = self.factor*di.dot(dj) + num_atoms_i.dot(num_atoms_j)

                        kernel_mat[n+i*n_dim:n+(i+1)*n_dim,j] = self.factor*ddi.dot(dj)
                        kernel_mat[i, m+j*n_dim:m+(j+1)*n_dim] = self.factor*di.dot(ddj.T)
                        kernel_mat[n+i*n_dim:n+(i+1)*n_dim,m+j*n_dim:m+(j+1)*n_dim] = self.factor*ddi.dot(ddj.T)
                return kernel_mat
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

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
