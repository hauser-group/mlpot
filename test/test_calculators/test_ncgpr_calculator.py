import unittest
import numpy as np

from ase.atoms import Atoms
from ase.calculators.emt import EMT
from mlpot.calculators.ncgprcalculator import NCGPRCalculator
from mlpot.kernels import RBFKernel


class GAPCalculatorTest(unittest.TestCase):
    def test_co_potential_curve(self):
        direction = np.array([1., 2., 3.])
        direction /= np.linalg.norm(direction)
        r_train = np.linspace(0.5, 7, 11)
        energies_train = []
        images_train = []
        for ri in r_train:
            image = Atoms(
                ['C', 'O'],
                positions=np.array([-0.5*ri*direction, 0.5*ri*direction]))
            image.set_calculator(EMT())
            energies_train.append(image.get_potential_energy())
            images_train.append(image)

        def to_radius(x):
            xyzs = x.get_positions()
            r = np.sqrt(np.sum((xyzs[1, :]-xyzs[0, :])**2))
            dr = np.zeros((1, 6))
            dr[0, 0] = (xyzs[0, 0] - xyzs[1, 0])/r
            dr[0, 1] = (xyzs[0, 1] - xyzs[1, 1])/r
            dr[0, 2] = (xyzs[0, 2] - xyzs[1, 2])/r
            dr[0, 3] = (xyzs[1, 0] - xyzs[0, 0])/r
            dr[0, 4] = (xyzs[1, 1] - xyzs[0, 1])/r
            dr[0, 5] = (xyzs[1, 2] - xyzs[0, 2])/r
            return [r], dr

        kernel = RBFKernel(constant=100.0, length_scale=1e-2)
        calc = NCGPRCalculator(input_transform=to_radius, kernel=kernel,
                               C1=1e8, C2=1e8, opt_restarts=0)

        [calc.add_data(im) for im in images_train]
        calc.fit()
        np.testing.assert_allclose(
            energies_train,
            [calc.predict(im)[0] for im in images_train])

    def test_build_diag(self):
        direction = np.array([1., 2., 3.])
        direction /= np.linalg.norm(direction)
        r_train = np.linspace(0.5, 7, 11)
        energies_train = []
        images_train = []
        for ri in r_train:
            image = Atoms(
                ['C', 'O'],
                positions=np.array([-0.5*ri*direction, 0.5*ri*direction]))
            image.set_calculator(EMT())
            energies_train.append(image.get_potential_energy())
            images_train.append(image)

        def to_radius(x):
            xyzs = x.get_positions()
            r = np.sqrt(np.sum((xyzs[1, :]-xyzs[0, :])**2))
            dr = np.zeros((1, 6))
            dr[0, 0] = (xyzs[0, 0] - xyzs[1, 0])/r
            dr[0, 1] = (xyzs[0, 1] - xyzs[1, 1])/r
            dr[0, 2] = (xyzs[0, 2] - xyzs[1, 2])/r
            dr[0, 3] = (xyzs[1, 0] - xyzs[0, 0])/r
            dr[0, 4] = (xyzs[1, 1] - xyzs[0, 1])/r
            dr[0, 5] = (xyzs[1, 2] - xyzs[0, 2])/r
            return [r], dr

        kernel = RBFKernel(constant=100.0, length_scale=1e-2)
        calc = NCGPRCalculator(input_transform=to_radius, kernel=kernel,
                               C1=1e8, C2=1e8, opt_restarts=0)

        [calc.add_data(im) for im in images_train]
        calc.fit()
        np.testing.assert_allclose(
            calc.build_kernel_diagonal((calc.q_train, calc.dq_train)),
            np.diag(calc.build_kernel_matrix()))

    def test_kernel_matrix_derivative(self):
        direction = np.array([1., 2., 3.])
        direction /= np.linalg.norm(direction)
        atoms = Atoms(
            ['C', 'O'],
            positions=np.array([-0.5*direction, 0.5*direction]))
        atoms.set_calculator(EMT())

        def to_radius(x):
            xyzs = x.get_positions()
            r = np.sqrt(np.sum((xyzs[1, :]-xyzs[0, :])**2))
            dr = np.zeros((1, 6))
            dr[0, 0] = (xyzs[0, 0] - xyzs[1, 0])/r
            dr[0, 1] = (xyzs[0, 1] - xyzs[1, 1])/r
            dr[0, 2] = (xyzs[0, 2] - xyzs[1, 2])/r
            dr[0, 3] = (xyzs[1, 0] - xyzs[0, 0])/r
            dr[0, 4] = (xyzs[1, 1] - xyzs[0, 1])/r
            dr[0, 5] = (xyzs[1, 2] - xyzs[0, 2])/r
            return [r], dr

        kernel = RBFKernel(constant=100.0, length_scale=0.23)
        calc = NCGPRCalculator(input_transform=to_radius, kernel=kernel,
                               C1=1e8, C2=1e8, opt_restarts=0)
        calc.add_data(atoms)
        K = calc.build_kernel_matrix()
        K_num = np.zeros_like(K)
        # kernel value is not tested:
        K_num[0, 0] = K[0, 0]
        x0 = atoms.get_positions()
        dx = 1e-4

        def num_dx_forth_order(fun, x0, y0, dx):
            return (8*fun(x0+dx, y0) - fun(x0+2*dx, y0)
                    - 8*fun(x0-dx, y0) + fun(x0-2*dx, y0)
                    )/(12*np.linalg.norm(dx))

        def num_dy_forth_order(fun, x0, y0, dy):
            return (8*fun(x0, y0+dy) - fun(x0, y0+2*dy)
                    - 8*fun(x0, y0-dy) + fun(x0, y0-2*dy)
                    )/(12*np.linalg.norm(dx))

        def num_dxdy_forth_order(fun, x0, y0, dx, dy):
            return (64*fun(x0+dx, y0+dy) - 8*fun(x0+dx, y0+2*dy)
                    - 64*fun(x0+dx, y0-dy) + 8*fun(x0+dx, y0-2*dy)
                    - 8*fun(x0+2*dx, y0+dy) + fun(x0+2*dx, y0+2*dy)
                    + 8*fun(x0+2*dx, y0-dy) - fun(x0+2*dx, y0-2*dy)
                    - 64*fun(x0-dx, y0+dy) + 8*fun(x0-dx, y0+2*dy)
                    + 64*fun(x0-dx, y0-dy) - 8*fun(x0-dx, y0-2*dy)
                    + 8*fun(x0-2*dx, y0+dy) - fun(x0-2*dx, y0+2*dy)
                    - 8*fun(x0-2*dx, y0-dy) + fun(x0-2*dx, y0-2*dy)
                    )/(144*np.linalg.norm(dx)*np.linalg.norm(dy))

        def K_fun(x, y):
            a = np.array([to_radius(Atoms(['C', 'O'], positions=x))[0]])
            b = np.array([to_radius(Atoms(['C', 'O'], positions=y))[0]])
            return calc.kernel(a, b, dx=True, dy=True)[0, 0]

        for i in range(6):
            dxi = np.zeros(6)
            dxi[i] = dx
            dxi = dxi.reshape((2, 3))
            # Test first derivative
            K_num[1+i, 0] = num_dx_forth_order(K_fun, x0, x0, dxi)

            for j in range(6):
                dxj = np.zeros(6)
                dxj[j] = dx
                dxj = dxj.reshape((2, 3))
                K_num[1+i, 1+j] = num_dxdy_forth_order(K_fun, x0, x0, dxi, dxj)

            # Test symmetry of derivatives
            K_num[0, 1+i] = num_dy_forth_order(K_fun, x0, x0, dxi)
        np.testing.assert_allclose(K, K_num, atol=1E-5)


if __name__ == '__main__':
    unittest.main()
