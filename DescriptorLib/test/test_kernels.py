import unittest
from ase.atoms import Atoms
import numpy as np
from DescriptorLib.kernels import RBFKernel, RBFKernel_with_factor, MaternKernel, DotProductKernel, NormalizedDotProductKernel

class KernelTest():
    class KernelTest(unittest.TestCase):
        def test_atomic_position_gradient(self):
            kernel = self.kernel(**self.kwargs)

            atomsX = np.random.randn(1,12)
            atomsY = np.random.randn(1,12)

            K = kernel(atomsX, atomsY, dx=True, dy=True)
            dK_num_total = np.zeros_like(K)
            dK_num_total[0,0] = K[0,0]

            dx = 1e-4
            for i in range(12):
                dxi = np.zeros(12)
                dxi[i] = dx
                K_plus = kernel(atomsX+dxi, atomsY, dx=True, dy=True)
                K_minus = kernel(atomsX-dxi, atomsY, dx=True, dy=True)
                # Test first derivative
                dK_num_total[1+i,0] = (K_plus[0,0] - K_minus[0,0])/(2*dx)
                # Approximate second derivative as numerical derivative of
                # first derivative
                dK_num_total[1+i,1:] = (K_plus[0,1:] - K_minus[0,1:])/(2*dx)

                # Test symmetry of derivatives
                K_plus = kernel(atomsX, atomsY+dxi, dx=True, dy=True)
                K_minus = kernel(atomsX, atomsY-dxi, dx=True, dy=True)
                dK_num_total[0,1+i] = (K_plus[0,0] - K_minus[0,0])/(2*dx)

            np.testing.assert_allclose(K, dK_num_total, atol=1E-9)

        def test_hyperparameter_gradient(self):
            kernel = self.kernel(**self.kwargs)
            atomsX = np.random.randn(5,12)
            atomsY = np.random.randn(5,12)

            K, dK = kernel(atomsX, atomsY, dx=True, dy=True, eval_gradient=True)
            dK_num = np.zeros_like(dK)

            # Derivative with respect to the hyperparameters:
            dt = 1e-6
            # Hyperparameters live on an exponential scale!!!
            for i in range(len(kernel.theta)):
                dti = np.zeros(len(kernel.theta))
                dti[i] = dt
                kernel.theta = np.log(np.exp(kernel.theta) + dti)
                K_plus = kernel(atomsX, atomsY, dx=True, dy=True, eval_gradient=False)
                kernel.theta = np.log(np.exp(kernel.theta) - 2*dti)
                K_minus = kernel(atomsX, atomsY, dx=True, dy=True, eval_gradient=False)
                kernel.theta = np.log(np.exp(kernel.theta) + dti)
                dK_num[:,:,i] = (K_plus - K_minus)/(2*dt)

            np.testing.assert_allclose(dK, dK_num, atol=1E-8)


class RBFKernelTest(KernelTest.KernelTest):
    kernel = RBFKernel
    kwargs = {'constant':23.4, 'factor':1.234, 'length_scale':1.321}

    def test_symmetry(self):
        #atomsX = [Atoms(['N','H','H','H'], positions = np.random.randn(4, 3))]
        atomsX = np.random.randn(1,12)
        #atomsY = [Atoms(['N','H','H','H'], positions = np.random.randn(4, 3))]
        atomsY = np.random.randn(1,12)
        constant = 23.4
        factor = 1.234
        length_scale = 1.321
        kernel = RBFKernel(
            constant=constant, factor=factor, length_scale=length_scale)

        K1 = kernel(atomsX, atomsY, dx=True, dy=True)
        K2 = kernel(atomsY, atomsX, dx=True, dy=True)

        N = len(atomsX)
        # K and H blocks are symmmetric
        np.testing.assert_allclose(K1[:N,:N], K2[:N,:N])
        np.testing.assert_allclose(K1[N:,N:], K2[N:,N:])
        # J and J^T are antisymmetric
        np.testing.assert_allclose(K1[:N,N:], -K2[:N,N:])
        np.testing.assert_allclose(K1[N:,:N], -K2[N:,:N])

    def test_equivalence_to_RBFKernel_with_factor(self):
        #atoms = []
        #for _ in range(10):
        #    atoms.append(Atoms(['H','O','H'],
        #        positions = np.random.randn(3, 3)))
        atoms = np.random.randn(10,9)

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
        np.testing.assert_allclose(dK_with[:,:,1:], dK_without)

class RBFKernelAnisoTest(KernelTest.KernelTest):
    kernel = RBFKernel
    kwargs = {'constant':23.4, 'factor':1.234, 'length_scale':np.array([1.321]*12)}


class RBFKernel_with_factorTest(KernelTest.KernelTest):
    kernel = RBFKernel_with_factor
    kwargs = {'constant':23.4, 'factor':1.234, 'length_scale':1.321}

class RBFKernel_with_factorAnisoTest(KernelTest.KernelTest):
    kernel = RBFKernel_with_factor
    kwargs = {'constant':23.4, 'factor':1.234, 'length_scale':np.array([1.321]*12)}    

class MaternKernelTest(KernelTest.KernelTest):
    kernel = MaternKernel
    kwargs = {'constant':23.4, 'factor':1.234, 'length_scale':1.321}

    def test_symmetry(self):
        #atomsX = [Atoms(['N','H','H','H'], positions = np.random.randn(4, 3))]
        atomsX = np.random.randn(1,12)
        #atomsY = [Atoms(['N','H','H','H'], positions = np.random.randn(4, 3))]
        atomsY = np.random.randn(1,12)

        constant = 23.4
        factor = 1.234
        length_scale = 1.321
        kernel = MaternKernel(
            constant=constant, factor=factor, length_scale=length_scale)

        K1 = kernel(atomsX, atomsY, dx=True, dy=True)
        K2 = kernel(atomsY, atomsX, dx=True, dy=True)

        N = len(atomsX)
        # K and H blocks are symmmetric
        np.testing.assert_allclose(K1[:N,:N], K2[:N,:N])
        np.testing.assert_allclose(K1[N:,N:], K2[N:,N:])
        # J and J^T are antisymmetric
        np.testing.assert_allclose(K1[:N,N:], -K2[:N,N:])
        np.testing.assert_allclose(K1[N:,:N], -K2[N:,:N])

# Still missing: Matern Anisotropic

class DotProductKernelTest(KernelTest.KernelTest):
    kernel = DotProductKernel
    kwargs = {'constant':23.4}

    def test_symmetry(self):
        #atomsX = [Atoms(['N','H','H','H'], positions = np.random.randn(4, 3))]
        atomsX = np.random.randn(1,12)
        #atomsY = [Atoms(['N','H','H','H'], positions = np.random.randn(4, 3))]
        atomsY = np.random.randn(1,12)

        constant = 23.4
        kernel = DotProductKernel(constant=constant)

        K1 = kernel(atomsX, atomsY, dx=True, dy=True)
        K2 = kernel(atomsY, atomsX, dx=True, dy=True)

        N = len(atomsX)
        # K and H blocks are symmmetric
        np.testing.assert_allclose(K1, K2.T)

class DotProductKernelExp1Test(KernelTest.KernelTest):
    kernel = DotProductKernel
    kwargs = {'constant':23.4, 'exponent':1}

class DotProductKernelExp4Test(KernelTest.KernelTest):
    kernel = DotProductKernel
    kwargs = {'constant':23.4, 'exponent':4}

class NormalizedDotProductKernelTest(KernelTest.KernelTest):
    kernel = NormalizedDotProductKernel
    kwargs = {'constant':1.234}

class NormalizedDotProductKernelExp1Test(KernelTest.KernelTest):
    kernel = NormalizedDotProductKernel
    kwargs = {'constant':1.234, 'exponent':1}

class NormalizedDotProductKernelExp4Test(KernelTest.KernelTest):
    kernel = NormalizedDotProductKernel
    kwargs = {'constant':1.234, 'exponent':4}

if __name__ == '__main__':
    unittest.main()
