import numpy as np
import unittest
from mlpot.kernels import Sum, ConstantKernel, RBFKernel


class SumTest(unittest.TestCase):

    def test_RBF_plus_float(self):
        n = 3
        m = 4
        n_dim = 12
        X = np.random.randn(n, n_dim)
        Y = np.random.randn(m, n_dim)

        constant = 10.0

        sum_kernel = RBFKernel() + constant
        ref_kernel = RBFKernel()

        K_sum, dK_sum = sum_kernel(X, Y, dx=True, dy=True, eval_gradient=True)
        K_ref, dK_ref1 = ref_kernel(X, Y, dx=True, dy=True, eval_gradient=True)
        K_ref[:n, :m] += constant
        # Derivative with respect to the second hyperparameter:
        dK_ref2 = np.zeros((n*(1+n_dim), m*(1+n_dim), 1))
        dK_ref2[:n, :m, 0] = 1.0

        np.testing.assert_allclose(K_sum, K_ref)
        np.testing.assert_allclose(dK_sum[:, :, :1], dK_ref1)
        np.testing.assert_allclose(dK_sum[:, :, -1:], dK_ref2)

    def test_RBF_plus_ConstantKernel(self):
        n = 3
        m = 4
        n_dim = 12
        X = np.random.randn(n, n_dim)
        Y = np.random.randn(m, n_dim)

        constant = 10.0

        sum_kernel = Sum(RBFKernel(), ConstantKernel(constant=constant))
        ref_kernel = RBFKernel()

        K_sum, dK_sum = sum_kernel(X, Y, dx=True, dy=True, eval_gradient=True)
        K_ref, dK_ref1 = ref_kernel(X, Y, dx=True, dy=True, eval_gradient=True)
        K_ref[:n, :m] += constant
        # Derivative with respect to the second hyperparameter:
        dK_ref2 = np.zeros((n*(1+n_dim), m*(1+n_dim), 1))
        dK_ref2[:n, :m, 0] = 1.0

        np.testing.assert_allclose(K_sum, K_ref)
        np.testing.assert_allclose(dK_sum[:, :, :1], dK_ref1)
        np.testing.assert_allclose(dK_sum[:, :, -1:], dK_ref2)


if __name__ == '__main__':
    unittest.main()
