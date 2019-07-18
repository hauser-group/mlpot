import unittest
import numpy as np
from mlpot.kernels import (ConstantKernel, RBFKernel, RBFKernel_with_factor,
                           MaternKernel, DotProductKernel,
                           NormalizedDotProductKernel,
                           NormalizedDotProductKernelwithHyperparameter)


class KernelTest():
    class KernelTest(unittest.TestCase):
        def test_atomic_position_gradient(self):
            atomsX = np.random.randn(1, 12)
            atomsY = np.random.randn(1, 12)

            K = self.kernel(atomsX, atomsY, dx=True, dy=True)
            dK_num_total = np.zeros_like(K)
            dK_num_total[0, 0] = K[0, 0]

            dx = 1e-4
            for i in range(12):
                dxi = np.zeros(12)
                dxi[i] = dx
                K_plus = self.kernel(atomsX+dxi, atomsY, dx=True, dy=True)
                K_minus = self.kernel(atomsX-dxi, atomsY, dx=True, dy=True)
                # Test first derivative
                dK_num_total[1+i, 0] = (K_plus[0, 0] - K_minus[0, 0])/(2*dx)
                # Approximate second derivative as numerical derivative of
                # first derivative
                dK_num_total[1+i, 1:] = (K_plus[0, 1:] - K_minus[0, 1:])/(2*dx)

                # Test symmetry of derivatives
                K_plus = self.kernel(atomsX, atomsY+dxi, dx=True, dy=True)
                K_minus = self.kernel(atomsX, atomsY-dxi, dx=True, dy=True)
                dK_num_total[0, 1+i] = (K_plus[0, 0] - K_minus[0, 0])/(2*dx)

            np.testing.assert_allclose(K, dK_num_total, atol=1E-9)

        def test_hyperparameter_gradient(self):
            atomsX = np.random.randn(5, 12)
            atomsY = np.random.randn(5, 12)

            K, dK = self.kernel(atomsX, atomsY, dx=True, dy=True,
                                eval_gradient=True)
            dK_num = np.zeros_like(dK)

            # Derivative with respect to the hyperparameters:
            dt = 1e-6
            # Hyperparameters live on an exponential scale!!!
            for i in range(len(self.kernel.theta)):
                dti = np.zeros(len(self.kernel.theta))
                dti[i] = dt
                self.kernel.theta = np.log(np.exp(self.kernel.theta) + dti)
                K_plus = self.kernel(atomsX, atomsY, dx=True, dy=True,
                                     eval_gradient=False)
                self.kernel.theta = np.log(np.exp(self.kernel.theta) - 2*dti)
                K_minus = self.kernel(atomsX, atomsY, dx=True, dy=True,
                                      eval_gradient=False)
                self.kernel.theta = np.log(np.exp(self.kernel.theta) + dti)
                dK_num[:, :, i] = (K_plus - K_minus)/(2*dt)

            np.testing.assert_allclose(dK, dK_num, atol=1E-8)

        def test_hyperparameter_gradient_XisY(self):
            """ In theory the gradient with respect to the hyperparameters
            should only ever be needed when the kernel is called with the same
            arguments. At least that is how scikit-learn implements kernels.
            """
            atomsX = np.random.randn(5, 12)

            K, dK = self.kernel(atomsX, atomsX, dx=True, dy=True,
                                eval_gradient=True)
            dK_num = np.zeros_like(dK)

            # Derivative with respect to the hyperparameters:
            dt = 1e-6
            # Hyperparameters live on an exponential scale!!!
            for i in range(len(self.kernel.theta)):
                dti = np.zeros(len(self.kernel.theta))
                dti[i] = dt
                self.kernel.theta = np.log(np.exp(self.kernel.theta) + dti)
                K_plus = self.kernel(atomsX, atomsX, dx=True, dy=True,
                                     eval_gradient=False)
                self.kernel.theta = np.log(np.exp(self.kernel.theta) - 2*dti)
                K_minus = self.kernel(atomsX, atomsX, dx=True, dy=True,
                                      eval_gradient=False)
                self.kernel.theta = np.log(np.exp(self.kernel.theta) + dti)
                dK_num[:, :, i] = (K_plus - K_minus)/(2*dt)

            np.testing.assert_allclose(dK, dK_num, atol=1E-8)

        def test_symmetry(self):
            atomsX = np.random.randn(1, 12)

            K = self.kernel(atomsX, atomsX, dx=True, dy=True)

            np.testing.assert_allclose(K, K.T)

        def test_diag(self):
            atomsX = np.random.randn(5, 12)

            K = self.kernel(atomsX, atomsX, dx=True, dy=True)
            diag_K = self.kernel.diag(atomsX)

            np.testing.assert_allclose(diag_K, np.diag(K), rtol=1E-6)


class ConstantKernelTest(KernelTest.KernelTest):
    kernel = ConstantKernel(constant=23.4)


class RBFKernelTest(KernelTest.KernelTest):
    kernel = RBFKernel(constant=23.4, factor=1.234, length_scale=1.321)

    def test_symmetry(self):
        atomsX = np.random.randn(1, 12)
        atomsY = np.random.randn(1, 12)
        constant = 23.4
        factor = 1.234
        length_scale = 1.321
        kernel = RBFKernel(
            constant=constant, factor=factor, length_scale=length_scale)

        K1 = kernel(atomsX, atomsY, dx=True, dy=True)
        K2 = kernel(atomsY, atomsX, dx=True, dy=True)

        N = len(atomsX)
        # K and H blocks are symmmetric
        np.testing.assert_allclose(K1[:N, :N], K2[:N, :N])
        np.testing.assert_allclose(K1[N:, N:], K2[N:, N:])
        # J and J^T are antisymmetric
        np.testing.assert_allclose(K1[:N, N:], -K2[:N, N:])
        np.testing.assert_allclose(K1[N:, :N], -K2[N:, :N])

    def test_equivalence_to_RBFKernel_with_factor(self):
        atoms = np.random.randn(10, 9)

        constant = 23.4
        factor = 1.234
        length_scale = 0.4321
        kernel_with = RBFKernel_with_factor(constant=constant, factor=factor,
                                            length_scale=length_scale)
        kernel_without = RBFKernel(constant=constant, factor=factor,
                                   length_scale=length_scale)
        K_with, dK_with = kernel_with(atoms, atoms, dx=True, dy=True,
                                      eval_gradient=True)
        K_without, dK_without = kernel_without(atoms, atoms, dx=True, dy=True,
                                               eval_gradient=True)
        np.testing.assert_allclose(K_with, K_without)
        np.testing.assert_allclose(dK_with[:, :, 1:], dK_without)


class RBFKernelAnisoTest(KernelTest.KernelTest):
    kernel = RBFKernel(constant=23.4, factor=1.234,
                       length_scale=np.array([1.321]*12))


class RBFKernel_with_factorTest(KernelTest.KernelTest):
    kernel = RBFKernel_with_factor(constant=23.4, factor=1.234,
                                   length_scale=1.321)


class RBFKernel_with_factorAnisoTest(KernelTest.KernelTest):
    kernel = RBFKernel_with_factor(constant=23.4, factor=1.234,
                                   length_scale=np.array([1.321]*12))


class MaternKernelTest(KernelTest.KernelTest):
    kernel = MaternKernel(constant=23.4, factor=1.234, length_scale=1.321)

    def test_symmetry(self):
        atomsX = np.random.randn(1, 12)
        atomsY = np.random.randn(1, 12)

        constant = 23.4
        factor = 1.234
        length_scale = 1.321
        kernel = MaternKernel(
            constant=constant, factor=factor, length_scale=length_scale)

        K1 = kernel(atomsX, atomsY, dx=True, dy=True)
        K2 = kernel(atomsY, atomsX, dx=True, dy=True)

        N = len(atomsX)
        # K and H blocks are symmmetric
        np.testing.assert_allclose(K1[:N, :N], K2[:N, :N])
        np.testing.assert_allclose(K1[N:, N:], K2[N:, N:])
        # J and J^T are antisymmetric
        np.testing.assert_allclose(K1[:N, N:], -K2[:N, N:])
        np.testing.assert_allclose(K1[N:, :N], -K2[N:, :N])


class MaternKernelAnisoTest(KernelTest.KernelTest):
    kernel = MaternKernel(constant=23.4, factor=1.234,
                          length_scale=np.array([1.321]*12))


class DotProductKernelTest(KernelTest.KernelTest):
    kernel = DotProductKernel(sigma0=2.34)

    def test_symmetry(self):
        atomsX = np.random.randn(1, 12)
        atomsY = np.random.randn(1, 12)

        sigma0 = 23.4
        kernel = DotProductKernel(sigma0=sigma0)

        K1 = kernel(atomsX, atomsY, dx=True, dy=True)
        K2 = kernel(atomsY, atomsX, dx=True, dy=True)

        # K and H blocks are symmmetric
        np.testing.assert_allclose(K1, K2.T)


class DotProductKernelExp1Test(KernelTest.KernelTest):
    kernel = DotProductKernel(sigma0=2.34, exponent=1)


class DotProductKernelExp4Test(KernelTest.KernelTest):
    kernel = DotProductKernel(sigma0=2.34, exponent=4)


class NormalizedDotProductKernelTest(KernelTest.KernelTest):
    kernel = NormalizedDotProductKernel(sigma0=0.321, constant=1.234)


class NormalizedDotProductKernelExp1Test(KernelTest.KernelTest):
    kernel = NormalizedDotProductKernel(sigma0=0.321, constant=1.234,
                                        exponent=1)


class NormalizedDotProductKernelExp4Test(KernelTest.KernelTest):
    kernel = NormalizedDotProductKernel(sigma0=0.321, constant=1.234,
                                        exponent=4)


class NormalizedDotProductKernelwithHyperparameterTest(KernelTest.KernelTest):
    kernel = NormalizedDotProductKernelwithHyperparameter(
        sigma0=0.321, exponent=2, constant=1.234
    )


class NormalizedDotProductKernelwithHyperparameterExp4Test(
        KernelTest.KernelTest):
    kernel = NormalizedDotProductKernelwithHyperparameter(
        sigma0=0.321, exponent=4, constant=1.234
    )


if __name__ == '__main__':
    unittest.main()
