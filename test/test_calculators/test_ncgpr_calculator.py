import unittest
import numpy as np

from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from mlpot.calculators.ncgprcalculator import NCGPRCalculator
from mlpot.calculators.gprcalculator import GPRCalculator
from mlpot.geometry import to_primitives_factory
from mlpot.kernels import RBFKernel


class GAPCalculatorTest(unittest.TestCase):

    @staticmethod
    def num_dx_forth_order(fun, x0, y0, dx):
        return (8*fun(x0+dx, y0) - fun(x0+2*dx, y0)
                - 8*fun(x0-dx, y0) + fun(x0-2*dx, y0)
                )/(12*np.linalg.norm(dx))

    @staticmethod
    def num_dy_forth_order(fun, x0, y0, dy):
        return (8*fun(x0, y0+dy) - fun(x0, y0+2*dy)
                - 8*fun(x0, y0-dy) + fun(x0, y0-2*dy)
                )/(12*np.linalg.norm(dy))

    @staticmethod
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

    def test_equivalence_to_GPR(self):

        def to_cartesian(atoms):
            xyzs = atoms.get_positions().flatten()
            return xyzs, np.eye(len(xyzs))

        gpr_calc = GPRCalculator(
            kernel=RBFKernel(constant=100.0, length_scale=0.456), C1=1e8,
            C2=1e8, opt_restarts=1)
        ncgpr_calc = NCGPRCalculator(
            kernel=RBFKernel(constant=100.0, length_scale=0.456),
            input_transform=to_cartesian, C1=1e8, C2=1e8, opt_restarts=1)

        atoms = molecule('cyclobutane')
        atoms.set_calculator(EMT())
        xyzs = atoms.get_positions()
        gpr_calc.add_data(atoms)
        ncgpr_calc.add_data(atoms)
        for i in range(8):
            a = atoms.copy()
            a.set_calculator(EMT())
            a.set_positions(xyzs + 0.5*np.random.randn(*xyzs.shape))
            gpr_calc.add_data(a)
            ncgpr_calc.add_data(a)

        np.testing.assert_allclose(ncgpr_calc.build_kernel_matrix(),
                                   gpr_calc.build_kernel_matrix())

        gpr_calc.fit()
        ncgpr_calc.fit()
        np.testing.assert_allclose(ncgpr_calc.kernel.theta,
                                   gpr_calc.kernel.theta)
        np.testing.assert_allclose(ncgpr_calc.alpha, gpr_calc.alpha)

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

        kernel = RBFKernel(constant=100.0, length_scale=1e-1)
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

        kernel = RBFKernel(constant=100.0, length_scale=1e-1)
        calc = NCGPRCalculator(input_transform=to_radius, kernel=kernel,
                               C1=1e8, C2=1e8, opt_restarts=0)

        [calc.add_data(im) for im in images_train]
        calc.fit()
        np.testing.assert_allclose(
            calc.build_kernel_diagonal((calc.q_train, calc.dq_train)),
            np.diag(calc.build_kernel_matrix()))

    def test_co_kernel_derivative(self):
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

        def K_fun(x, y):
            a = np.array([to_radius(Atoms(['C', 'O'], positions=x))[0]])
            b = np.array([to_radius(Atoms(['C', 'O'], positions=y))[0]])
            return calc.kernel(a, b, dx=True, dy=True)[0, 0]

        for i in range(6):
            dxi = np.zeros(6)
            dxi[i] = dx
            dxi = dxi.reshape((2, 3))
            # Test first derivative
            K_num[1+i, 0] = self.num_dx_forth_order(K_fun, x0, x0, dxi)

            for j in range(6):
                dxj = np.zeros(6)
                dxj[j] = dx
                dxj = dxj.reshape((2, 3))
                K_num[1+i, 1+j] = self.num_dxdy_forth_order(
                    K_fun, x0, x0, dxi, dxj)

            # Test symmetry of derivatives
            K_num[0, 1+i] = self.num_dy_forth_order(K_fun, x0, x0, dxi)
        np.testing.assert_allclose(K, K_num, atol=1E-5)

    def test_ethane_primitives_kernel_derivative(self):
        atoms = molecule('C2H6')
        atoms.set_calculator(EMT())
        # Add gaussian noise because of numerical problem for
        # the 180 degree angle
        x0 = atoms.get_positions() + 1e-3*np.random.randn(8, 3)
        atoms.set_positions(x0)
        symbols = atoms.get_chemical_symbols()
        # Ethane bonds:
        bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
        transform, _, _ = to_primitives_factory(bonds)
        kernel = RBFKernel(constant=100.0, length_scale=1.23)
        calc = NCGPRCalculator(input_transform=transform, kernel=kernel,
                               C1=1e8, C2=1e8, opt_restarts=0)
        calc.add_data(atoms)
        K = calc.build_kernel_matrix()

        K_num = np.zeros_like(K)
        # kernel value is not tested:
        K_num[0, 0] = K[0, 0]
        dx = 1e-4

        def K_fun(x, y):
            qx, dqx = transform(Atoms(symbols, positions=x))
            calc.q_train, calc.dq_train = [qx], [dqx]
            return calc.build_kernel_matrix(
                X_star=transform(Atoms(symbols, positions=y)))[0, 0]

        def K_dY(x, y):
            qx, dqx = transform(Atoms(symbols, positions=x))
            calc.q_train, calc.dq_train = [qx], [dqx]
            return calc.build_kernel_matrix(
                X_star=transform(Atoms(symbols, positions=y)))[0, 1:]

        for i in range(len(atoms)*3):
            dxi = np.zeros((len(atoms), 3))
            dxi.flat[i] = dx
            # Test first derivative
            K_num[1+i, 0] = self.num_dx_forth_order(K_fun, x0, x0, dxi)
            # Approximate second derivative as numerical derivative of
            # analytical first derivative
            K_num[1+i, 1:] = self.num_dx_forth_order(K_dY, x0, x0, dxi)
            # for j in range(len(atoms)*3):
            #     dxj = np.zeros((len(atoms), 3))
            #     dxj.flat[j] = dx
            #     K_num[1+i, 1+j] = self.num_dxdy_forth_order(
            #         K_fun, x0, x0, dxi, dxj)

            # Test symmetry of derivatives
            K_num[0, 1+i] = self.num_dy_forth_order(K_fun, x0, x0, dxi)
        np.testing.assert_allclose(K, K_num, atol=1E-5)


if __name__ == '__main__':
    unittest.main()
