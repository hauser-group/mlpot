import numpy as np
import unittest
from mlpot.kernels import (Sum, Product, Rescaling, Exponentiation,
                           ConstantKernel, RBFKernel,
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
        np.testing.assert_allclose(dK_sum[:, :, :-1], dK_ref1)
        np.testing.assert_allclose(dK_sum[:, :, -1:], dK_ref2)


class RBFplusRBFTest(KernelTest.KernelTest):
    kernel = Sum(RBFKernel(length_scale=0.8),
                 RBFKernel(length_scale=1.2))


class RBFtimesConstantTest(KernelTest.KernelTest):
    kernel = Product(RBFKernel(), ConstantKernel(constant=10.0))

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
        np.testing.assert_allclose(dK_prod[:, :, :-1], dK_ref1)
        np.testing.assert_allclose(dK_prod[:, :, -1:], dK_ref2)


class DottimesConstantTest(KernelTest.KernelTest):
    kernel = Product(DotProductKernel(exponent=2),
                     ConstantKernel(constant=10.0))


class RBFtimesRBFTest(KernelTest.KernelTest):
    kernel = Product(RBFKernel(length_scale=0.8),
                     RBFKernel(length_scale=1.2))


class DottimeDotTest(KernelTest.KernelTest):
    kernel = Product(DotProductKernel(sigma0=0.1),
                     DotProductKernel(sigma0=0.1))

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


class RescalingTest(KernelTest.KernelTest):
    kernel = Rescaling(RBFKernel(), 10.0)

    def test_comparison2product(self):
        n = 3
        m = 4
        n_dim = 12
        X = np.random.randn(n, n_dim)
        Y = np.random.randn(m, n_dim)

        factor = 10.0

        kernel1 = Rescaling(RBFKernel(), factor)
        kernel2 = Product(RBFKernel(), ConstantKernel(constant=factor))

        K1, dK1 = kernel1(X, Y, dx=True, dy=True, eval_gradient=True)
        K2, dK2 = kernel2(X, Y, dx=True, dy=True, eval_gradient=True)

        np.testing.assert_allclose(K1, K2)
        np.testing.assert_allclose(dK1, dK2)

    def test__rmul__(self):
        X = np.random.randn(3, 4)
        Y = np.random.randn(3, 4)

        factor = 10.0

        kernel1 = factor * RBFKernel()
        kernel2 = ConstantKernel(constant=factor) * RBFKernel()
        np.testing.assert_equal(kernel1.theta, kernel2.theta)

        K1, dK1 = kernel1(X, Y, dx=True, dy=True, eval_gradient=True)
        K2, dK2 = kernel2(X, Y, dx=True, dy=True, eval_gradient=True)

        np.testing.assert_allclose(K1, K2)
        np.testing.assert_allclose(dK1, dK2)

    def test__mul__(self):
        X = np.random.randn(3, 4)
        Y = np.random.randn(3, 4)

        factor = 10.0

        kernel1 = RBFKernel() * factor
        kernel2 = RBFKernel() * ConstantKernel(constant=factor)
        np.testing.assert_equal(kernel1.theta, kernel2.theta)

        K1, dK1 = kernel1(X, Y, dx=True, dy=True, eval_gradient=True)
        K2, dK2 = kernel2(X, Y, dx=True, dy=True, eval_gradient=True)

        np.testing.assert_allclose(K1, K2)
        np.testing.assert_allclose(dK1, dK2)


class ExponentiationTest(KernelTest.KernelTest):
    kernel = Exponentiation(RBFKernel(), 2.0)

    def test_rbf_squared(self):
        X = np.random.randn(3, 4)
        Y = np.random.randn(3, 4)

        kernel1 = Exponentiation(RBFKernel(length_scale=2.0), exponent=4)
        kernel2 = RBFKernel(length_scale=1.0)

        K1, dK1 = kernel1(X, Y, dx=True, dy=True, eval_gradient=True)
        K2, dK2 = kernel2(X, Y, dx=True, dy=True, eval_gradient=True)

        np.testing.assert_allclose(K1, K2)
        np.testing.assert_allclose(2*dK1, dK2)

    def test_dot_product(self):
        X = np.random.randn(3, 4)
        Y = np.random.randn(3, 4)

        kernel1 = Exponentiation(DotProductKernel(exponent=2), exponent=2)
        kernel2 = DotProductKernel(exponent=4)

        K1 = kernel1(X, Y, dx=True, dy=True)
        K2 = kernel2(X, Y, dx=True, dy=True)

        np.testing.assert_allclose(K1, K2)

    def test_comparison2product(self):
        X = np.random.randn(3, 4)
        Y = np.random.randn(3, 4)

        kernel1 = Exponentiation(RBFKernel(length_scale=1.23), exponent=2)
        kernel2 = Product(RBFKernel(length_scale=1.23),
                          RBFKernel(length_scale=1.23))

        K1 = kernel1(X, Y, dx=True, dy=True)
        K2 = kernel2(X, Y, dx=True, dy=True)

        np.testing.assert_allclose(K1, K2)


if __name__ == '__main__':
    unittest.main()
