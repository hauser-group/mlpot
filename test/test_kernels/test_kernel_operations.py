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

    def test_atomic_position_gradient(self):
        kernel = Sum(RBFKernel(), ConstantKernel(constant=10))

        atomsX = np.random.randn(1, 12)
        atomsY = np.random.randn(1, 12)

        K = kernel(atomsX, atomsY, dx=True, dy=True)
        dK_num_total = np.zeros_like(K)
        dK_num_total[0, 0] = K[0, 0]

        dx = 1e-4
        for i in range(12):
            dxi = np.zeros(12)
            dxi[i] = dx
            K_plus = kernel(atomsX+dxi, atomsY, dx=True, dy=True)
            K_minus = kernel(atomsX-dxi, atomsY, dx=True, dy=True)
            # Test first derivative
            dK_num_total[1+i, 0] = (K_plus[0, 0] - K_minus[0, 0])/(2*dx)
            # Approximate second derivative as numerical derivative of
            # first derivative
            dK_num_total[1+i, 1:] = (K_plus[0, 1:] - K_minus[0, 1:])/(2*dx)

            # Test symmetry of derivatives
            K_plus = kernel(atomsX, atomsY+dxi, dx=True, dy=True)
            K_minus = kernel(atomsX, atomsY-dxi, dx=True, dy=True)
            dK_num_total[0, 1+i] = (K_plus[0, 0] - K_minus[0, 0])/(2*dx)

        np.testing.assert_allclose(K, dK_num_total, atol=1E-9)

    def test_hyperparameter_gradient(self):
        kernel = Sum(RBFKernel(), ConstantKernel(constant=10.0))
        atomsX = np.random.randn(5, 12)
        atomsY = np.random.randn(5, 12)

        K, dK = kernel(atomsX, atomsY, dx=True, dy=True,
                       eval_gradient=True)
        dK_num = np.zeros_like(dK)

        # Derivative with respect to the hyperparameters:
        dt = 1e-6
        # Hyperparameters live on an exponential scale!!!
        for i in range(len(kernel.theta)):
            dti = np.zeros(len(kernel.theta))
            dti[i] = dt
            kernel.theta = np.log(np.exp(kernel.theta) + dti)
            K_plus = kernel(atomsX, atomsY, dx=True, dy=True,
                            eval_gradient=False)
            kernel.theta = np.log(np.exp(kernel.theta) - 2*dti)
            K_minus = kernel(atomsX, atomsY, dx=True, dy=True,
                             eval_gradient=False)
            kernel.theta = np.log(np.exp(kernel.theta) + dti)
            dK_num[:, :, i] = (K_plus - K_minus)/(2*dt)

        np.testing.assert_allclose(dK, dK_num, atol=1E-8)


if __name__ == '__main__':
    unittest.main()
