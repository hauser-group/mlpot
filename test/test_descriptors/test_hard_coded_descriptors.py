import unittest
from mlpot.descriptors import DescriptorSet
import numpy as np


class HardCodedTest(unittest.TestCase):

    def test_G4_trimer(self):
        with DescriptorSet(['Ag'], cutoff=8.0) as ds1:
            with DescriptorSet(['Ag'], cutoff=8.0) as ds2:
                types = ['Ag', 'Ag', 'Ag']

                ang_etas = np.array([0.0001, 0.003, 0.008])
                zetas = np.array([1.0, 4.0])
                for ang_eta in ang_etas:
                    for lamb in [-1.0, 1.0]:
                        for zeta in zetas:
                            ds1.add_three_body_descriptor(
                                'Ag', 'Ag', 'Ag', 'BehlerG4',
                                [lamb, zeta, ang_eta], cuttype='cos')
                            ds2.add_three_body_descriptor(
                                'Ag', 'Ag', 'Ag', 'BehlerG4auto',
                                [lamb, zeta, ang_eta], cuttype='cos')

                N = 30
                r_vec = np.linspace(1., 5., N)
                theta_vec = np.linspace(0.3*np.pi, 2.*np.pi, N, endpoint=True)
                for ri in r_vec:
                    for ti in theta_vec:
                        xyzs = np.array([[0.0, 0.0, 0.0],
                                        [0.5*ri, 0.0, 0.0],
                                        [ri*np.cos(ti), ri*np.sin(ti), 0.0]])
                        print(ti, 0.01*np.pi, np.cos(ti), np.sin(ti),
                              np.cos(ti)**2 + np.sin(ti)**2)
                        print(xyzs)

                        G1 = ds1.eval_atomwise(types, xyzs)
                        G2 = ds2.eval_atomwise(types, xyzs)
                        np.testing.assert_allclose(G1, G2, equal_nan=False)

                        dG1 = ds1.eval_derivatives_atomwise(types, xyzs)
                        dG2 = ds2.eval_derivatives_atomwise(types, xyzs)
                        print(dG1[0][0])
                        print(dG2[0][0])
                        np.testing.assert_allclose(dG1, dG2,
                                                   equal_nan=False, atol=1e-15)

                        G1, dG1 = (
                            ds1.eval_with_derivatives_atomwise(types, xyzs))
                        G2, dG2 = (
                            ds2.eval_with_derivatives_atomwise(types, xyzs))
                        #np.testing.assert_allclose(G1, G2, equal_nan=False)
                        #np.testing.assert_allclose(dG1, dG2,
                        #                           equal_nan=False, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
