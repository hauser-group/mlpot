import numpy as np
import unittest
from ase.build import molecule
from mlpot.geometry import to_primitives_factory, dist, angle, dihedral


class GeometryToolsTest(unittest.TestCase):

    def ethane_z_matrix(self, atoms):
        xyzs = atoms.get_positions()

        r_CC, dr_CC = dist(xyzs, 1, 0, derivative=True)
        r_CH1, dr_CH1 = dist(xyzs, 2, 0, derivative=True)
        t_CH1, dt_CH1 = angle(xyzs, 2, 0, 1, derivative=True)

        r_CH2, dr_CH2 = dist(xyzs, 3, 0, derivative=True)
        t_CH2, dt_CH2 = angle(xyzs, 3, 0, 1, derivative=True)
        w_CH2, dw_CH2 = dihedral(xyzs, 3, 0, 1, 2, derivative=True)

        r_CH3, dr_CH3 = dist(xyzs, 4, 0, derivative=True)
        t_CH3, dt_CH3 = angle(xyzs, 4, 0, 1, derivative=True)
        w_CH3, dw_CH3 = dihedral(xyzs, 4, 0, 1, 2, derivative=True)

        r_CH4, dr_CH4 = dist(xyzs, 5, 1, derivative=True)
        t_CH4, dt_CH4 = angle(xyzs, 5, 1, 0, derivative=True)
        w_CH4, dw_CH4 = dihedral(xyzs, 5, 1, 0, 2, derivative=True)

        r_CH5, dr_CH5 = dist(xyzs, 6, 1, derivative=True)
        t_CH5, dt_CH5 = angle(xyzs, 6, 1, 0, derivative=True)
        w_CH5, dw_CH5 = dihedral(xyzs, 6, 1, 0, 2, derivative=True)

        r_CH6, dr_CH6 = dist(xyzs, 7, 1, derivative=True)
        t_CH6, dt_CH6 = angle(xyzs, 7, 1, 0, derivative=True)
        w_CH6, dw_CH6 = dihedral(xyzs, 7, 1, 0, 2, derivative=True)

        return (
            np.array([r_CC, r_CH1, r_CH2, r_CH3, r_CH4, r_CH5, r_CH6,
                      t_CH1, t_CH2, t_CH3, t_CH4, t_CH5, t_CH6,
                      w_CH2, w_CH3, w_CH4, w_CH5, w_CH6]),
            np.array([dr_CC, dr_CH1, dr_CH2, dr_CH3, dr_CH4, dr_CH5, dr_CH6,
                      dt_CH1, dt_CH2, dt_CH3, dt_CH4, dt_CH5, dt_CH6,
                      dw_CH2, dw_CH3, dw_CH4, dw_CH5, dw_CH6]))

    def test_internals_derivatives(self):
        atoms = molecule('C2H6')
        # Add gaussian noise because of numerical problem for
        # the 180 degree angle
        xyzs = atoms.get_positions() + 1e-3*np.random.randn(8, 3)
        atoms.set_positions(xyzs)
        q, dq = self.ethane_z_matrix(atoms)

        dq_num = np.zeros_like(dq)
        dx = 1e-5
        for i in range(len(xyzs)):
            for n in range(3):
                dxi = np.zeros_like(xyzs)
                dxi[i, n] = dx
                atoms.set_positions(xyzs + dxi)
                q_plus, _ = self.ethane_z_matrix(atoms)
                atoms.set_positions(xyzs - dxi)
                q_minus, _ = self.ethane_z_matrix(atoms)
                dq_num[:, 3*i+n] = (q_plus - q_minus)/(2*dx)

        np.testing.assert_allclose(dq, dq_num, atol=1e-8)

    def test_primitives_factory_ethane(self):
        atoms = molecule('C2H6')
        # Add gaussian noise because of numerical problem for
        # the 180 degree angle
        xyzs = atoms.get_positions() + 1e-3*np.random.randn(8, 3)
        atoms.set_positions(xyzs)
        # Ethane bonds:
        bonds = [(0, 1),
                 (0, 2),
                 (0, 3),
                 (0, 4),
                 (1, 5),
                 (1, 6),
                 (1, 7)]

        transform, angles, dihedrals = to_primitives_factory(bonds)

        # Compare to angles and dihedrals found by Avogadro
        C1, C2, H1, H2, H3, H4, H5, H6 = range(8)
        av_angles = [(C2, C1, H1), (C2, C1, H2), (C2, C1, H3),
                     (H1, C1, H2), (H1, C1, H3), (H2, C1, H3),
                     (H4, C2, H5), (H4, C2, H6), (C1, C2, H4),
                     (H5, C2, H6), (C1, C2, H5), (C1, C2, H6)]
        av_dihedrals = [(H4, C2, C1, H1), (H4, C2, C1, H2), (H4, C2, C1, H3),
                        (H5, C2, C1, H1), (H5, C2, C1, H2), (H5, C2, C1, H3),
                        (H6, C2, C1, H1), (H6, C2, C1, H2), (H6, C2, C1, H3)]
        self.assertEqual(len(angles), len(av_angles))
        self.assertEqual(len(dihedrals), len(av_dihedrals))

        for a in angles:
            self.assertTrue(a in av_angles or a[::-1] in av_angles)
        for d in dihedrals:
            self.assertTrue(d in av_dihedrals or d[::-1] in av_dihedrals)

        q, dq = transform(atoms)
        dq_num = np.zeros_like(dq)
        dx = 1e-5
        for i in range(len(xyzs)):
            for n in range(3):
                dxi = np.zeros_like(xyzs)
                dxi[i, n] = dx
                atoms.set_positions(xyzs + dxi)
                q_plus, _ = transform(atoms)
                atoms.set_positions(xyzs - dxi)
                q_minus, _ = transform(atoms)
                dq_num[:, 3*i+n] = (q_plus - q_minus)/(2*dx)

        np.testing.assert_allclose(dq, dq_num, atol=1e-8)

    def test_primitives_factory_cyclobutane(self):
        # Additional test case for a ring structure
        atoms = molecule('cyclobutane')
        xyzs = atoms.get_positions()
        # Ethane bonds:
        bonds = [(0, 2), (0, 3),
                 (1, 2), (1, 3),
                 (4, 0), (5, 0),
                 (6, 1), (7, 1),
                 (8, 2), (9, 2),
                 (10, 3), (11, 3)]

        transform, angles, dihedrals = to_primitives_factory(bonds)

        # Compare to angles and dihedrals found by Avogadro
        C1, C2, C3, C4, H1, H2, H3, H4, H5, H6, H7, H8 = range(12)
        av_angles = [(C3, C1, H1), (C4, C1, H1), (H1, C1, H2),
                     (C3, C1, C4), (C3, C1, H2), (C4, C1, H2),
                     (C3, C2, H3), (C4, C2, H3), (H3, C2, H4),
                     (C3, C2, C4), (C3, C2, H4), (C4, C2, H4),
                     (C1, C3, H6), (C2, C3, H6), (H5, C3, H6),
                     (C1, C3, C2), (C1, C3, H5), (C2, C3, H5),
                     (C1, C4, H8), (C2, C4, H8), (H7, C4, H8),
                     (C1, C4, C2), (C1, C4, H7), (C2, C4, H7)]
        av_dihedrals = [(H6, C3, C1, H1), (H6, C3, C1, C4), (H6, C3, C1, H2),
                        (C2, C3, C1, H1), (C2, C3, C1, C4), (C2, C3, C1, H2),
                        (H5, C3, C1, H1), (H5, C3, C1, C4), (H5, C3, C1, H2),
                        (H6, C3, C2, H3), (H6, C3, C2, C4), (H6, C3, C2, H4),
                        (C1, C3, C2, H3), (C1, C3, C2, C4), (C1, C3, C2, H4),
                        (H5, C3, C2, H3), (H5, C3, C2, C4), (H5, C3, C2, H4),
                        (H8, C4, C1, H1), (H8, C4, C1, C3), (H8, C4, C1, H2),
                        (C2, C4, C1, H1), (C2, C4, C1, C3), (C2, C4, C1, H2),
                        (H7, C4, C1, H1), (H7, C4, C1, C3), (H7, C4, C1, H2),
                        (H8, C4, C2, H3), (H8, C4, C2, C3), (H8, C4, C2, H4),
                        (C1, C4, C2, H3), (C1, C4, C2, C3), (C1, C4, C2, H4),
                        (H7, C4, C2, H3), (H7, C4, C2, C3), (H7, C4, C2, H4)]
        self.assertEqual(len(angles), len(av_angles))
        self.assertEqual(len(dihedrals), len(av_dihedrals))

        for a in angles:
            self.assertTrue(a in av_angles or a[::-1] in av_angles)
        for d in dihedrals:
            self.assertTrue(d in av_dihedrals or d[::-1] in av_dihedrals)

        q, dq = transform(atoms)
        dq_num = np.zeros_like(dq)
        dx = 1e-5
        for i in range(len(xyzs)):
            for n in range(3):
                dxi = np.zeros_like(xyzs)
                dxi[i, n] = dx
                atoms.set_positions(xyzs + dxi)
                q_plus, _ = transform(atoms)
                atoms.set_positions(xyzs - dxi)
                q_minus, _ = transform(atoms)
                dq_num[:, 3*i+n] = (q_plus - q_minus)/(2*dx)

        np.testing.assert_allclose(dq, dq_num, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
