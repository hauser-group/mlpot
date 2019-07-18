import numpy as np
import unittest
from mlpot.kernels import (Sum, Product, ConstantKernel, RBFKernel,
                           DotProductKernel)
try:
    from test_kernels import KernelTest
except ImportError:
    from .test_kernels import KernelTest


class RBFplusConstantTest(KernelTest.KernelTest):
    kernel = Sum(RBFKernel(), ConstantKernel(constant=10.0))

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


class RBFplusRBFTest(KernelTest.KernelTest):
    kernel = Sum(RBFKernel(length_scale=0.8),
                 RBFKernel(length_scale=1.2))


class RBFtimesConstantTest(KernelTest.KernelTest):
    kernel = Product(RBFKernel(), ConstantKernel(constant=10.0))

    def test_RBF_times_float(self):
        n = 3
        m = 4
        n_dim = 12
        X = np.random.randn(n, n_dim)
        Y = np.random.randn(m, n_dim)

        factor = 10.0

        prod_kernel = RBFKernel() * factor
        ref_kernel = RBFKernel()

        K_prod, dK_prod = prod_kernel(
            X, Y, dx=True, dy=True, eval_gradient=True)
        K_ref, dK_ref1 = ref_kernel(X, Y, dx=True, dy=True, eval_gradient=True)
        # Derivative with respect to the second hyperparameter:
        dK_ref2 = np.zeros((n*(1+n_dim), m*(1+n_dim), 1))
        dK_ref2[:, :, 0] = K_ref

        K_ref *= factor
        dK_ref1 *= factor

        np.testing.assert_allclose(K_prod, K_ref)
        np.testing.assert_allclose(dK_prod[:, :, :1], dK_ref1)
        np.testing.assert_allclose(dK_prod[:, :, -1:], dK_ref2)

    def test_RBF_times_ConstantKernel(self):
        n = 3
        m = 4
        n_dim = 12
        X = np.random.randn(n, n_dim)
        Y = np.random.randn(m, n_dim)

        factor = 10.0

        prod_kernel = Product(RBFKernel(), ConstantKernel(constant=factor))
        ref_kernel = RBFKernel()

        K_prod, dK_prod = prod_kernel(
            X, Y, dx=True, dy=True, eval_gradient=True)
        K_ref, dK_ref1 = ref_kernel(X, Y, dx=True, dy=True, eval_gradient=True)
        # Derivative with respect to the second hyperparameter:
        dK_ref2 = np.zeros((n*(1+n_dim), m*(1+n_dim), 1))
        dK_ref2[:, :, 0] = K_ref

        K_ref *= factor
        dK_ref1 *= factor

        np.testing.assert_allclose(K_prod, K_ref)
        np.testing.assert_allclose(dK_prod[:, :, :1], dK_ref1)
        np.testing.assert_allclose(dK_prod[:, :, -1:], dK_ref2)


class DottimesConstantTest(KernelTest.KernelTest):
    kernel = Product(DotProductKernel(exponent=2),
                     ConstantKernel(constant=10.0))


class RBFtimesRBFTest(KernelTest.KernelTest):
    kernel = Product(RBFKernel(length_scale=0.8),
                     RBFKernel(length_scale=1.2))


class DottimeDotTest(KernelTest.KernelTest):
    kernel = Product(DotProductKernel(), DotProductKernel())

    def test_2times2is4(self):
        n = 3
        m = 4
        n_dim = 12
        X = np.random.randn(n, n_dim)
        Y = np.random.randn(m, n_dim)

        kernel1 = Product(DotProductKernel(), DotProductKernel())
        kernel2 = DotProductKernel(exponent=4)

        K1 = kernel1(X, Y, dx=True, dy=True)
        K2 = kernel2(X, Y, dx=True, dy=True)

        np.testing.assert_allclose(K1, K2)


if __name__ == '__main__':
    unittest.main()
