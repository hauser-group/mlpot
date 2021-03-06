import unittest
from mlpot.descriptors import DescriptorSet
import numpy as np


class AgTrimerTest(unittest.TestCase):

    def test_trimer(self):
        with DescriptorSet(['Ag'], cutoff=8.0) as ds:
            types = ['Ag', 'Ag', 'Ag']

            def fcut(r):
                return 0.5*(1.0 + np.cos(np.pi*r/ds.cutoff))*(r < ds.cutoff)

            # Parameters from Artrith and Kolpak Nano Lett. 2014, 14, 2670
            etas = np.array([0.0009, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2])
            for eta in etas:
                ds.add_two_body_descriptor('Ag', 'Ag', 'BehlerG2', [eta, 0.0],
                                           cuttype='cos')

            ang_etas = np.array([0.0001, 0.003, 0.008])
            zetas = np.array([1.0, 4.0])
            for ang_eta in ang_etas:
                for lamb in [-1.0, 1.0]:
                    for zeta in zetas:
                        ds.add_three_body_descriptor(
                            'Ag', 'Ag', 'Ag', 'BehlerG4',
                            [lamb, zeta, ang_eta], cuttype='cos')

            # Also test BehlerG5
            for ang_eta in ang_etas:
                for lamb in [-1.0, 1.0]:
                    for zeta in zetas:
                        ds.add_three_body_descriptor(
                            'Ag', 'Ag', 'Ag', 'BehlerG5',
                            [lamb, zeta, ang_eta], cuttype='cos')

            N = 30
            r_vec = np.linspace(1., 5., N)
            theta_vec = np.linspace(0.0*np.pi, 2.*np.pi, N, endpoint=True)
            for ri in r_vec:
                for ti in theta_vec:
                    xyzs = np.array([[0.0, 0.0, 0.0],
                                    [0.5*ri, 0.0, 0.0],
                                    [ri*np.cos(ti), ri*np.sin(ti), 0.0]])

                    rij = np.linalg.norm(xyzs[0, :]-xyzs[1, :])
                    rik = np.linalg.norm(xyzs[0, :]-xyzs[2, :])
                    rjk = np.linalg.norm(xyzs[1, :]-xyzs[2, :])
                    np.testing.assert_allclose(
                        rjk,
                        np.sqrt(rij**2+rik**2-2.*rij*rik*np.cos(ti)),
                        atol=1E-12)

                    Gs = ds.eval(types, xyzs)
                    Gs_atomwise = ds.eval_atomwise(types, xyzs)
                    Gs_ref = np.concatenate(
                        [np.exp(-etas*rij**2)*fcut(rij)
                         + np.exp(-etas*rik**2)*fcut(rik)] +
                        [2**(1.-zetas)*np.exp(-eta*(rij**2+rik**2+rjk**2)) *
                         (1.+lamb*np.cos(ti))**zetas *
                         fcut(rij)*fcut(rik)*fcut(rjk)
                         for eta in ang_etas for lamb in [-1.0, 1.0]] +
                        [2**(1.-zetas)*np.exp(-eta*(rij**2+rik**2)) *
                         (1.+lamb*np.cos(ti))**zetas*fcut(rij)*fcut(rik)
                         for eta in ang_etas for lamb in [-1.0, 1.0]])
                    np.testing.assert_allclose(Gs, Gs_atomwise,
                                               equal_nan=False)
                    np.testing.assert_allclose(Gs[0], Gs_ref,
                                               equal_nan=False)
                    np.testing.assert_allclose(Gs_atomwise[0], Gs_ref,
                                               equal_nan=False)

                    dGs = ds.eval_derivatives(types, xyzs)
                    dGs_atomwise = ds.eval_derivatives_atomwise(types, xyzs)
                    # Adding the equal_nan=False option shows a bug for
                    # descriptors using rik as input
                    np.testing.assert_allclose(dGs, dGs_atomwise,
                                               equal_nan=False)

                    Gs, dGs = ds.eval_with_derivatives(types, xyzs)
                    Gs_atomwise, dGs_atomwise = (
                        ds.eval_with_derivatives_atomwise(types, xyzs))
                    np.testing.assert_allclose(Gs, Gs_atomwise,
                                               equal_nan=False)
                    np.testing.assert_allclose(dGs, dGs_atomwise,
                                               equal_nan=False, atol=1e-15)

    def test_equidistant_trimer(self):
        with DescriptorSet(['Ag'], cutoff=8.0) as ds:
            types = ['Ag', 'Ag', 'Ag']

            N = 30
            r_vec = np.linspace(1., 5., N)
            theta_vec = np.linspace(0.0*np.pi, 2.*np.pi, N, endpoint=True)
            for ri in r_vec:
                for ti in theta_vec:
                    xyzs = np.array([[0.0, 0.0, 0.0],
                                    [ri, 0.0, 0.0],
                                    [ri*np.cos(ti), ri*np.sin(ti), 0.0]])

                    Gs = ds.eval(types, xyzs)
                    Gs_atomwise = ds.eval_atomwise(types, xyzs)
                    np.testing.assert_allclose(Gs, Gs_atomwise,
                                               equal_nan=False)

                    dGs = ds.eval_derivatives(types, xyzs)
                    dGs_atomwise = ds.eval_derivatives_atomwise(types, xyzs)
                    np.testing.assert_allclose(dGs, dGs_atomwise,
                                               equal_nan=False)

                    Gs, dGs = ds.eval_with_derivatives(types, xyzs)
                    Gs_atomwise, dGs_atomwise = (
                        ds.eval_with_derivatives_atomwise(types, xyzs))
                    np.testing.assert_allclose(Gs, Gs_atomwise,
                                               equal_nan=False)
                    np.testing.assert_allclose(dGs, dGs_atomwise,
                                               equal_nan=False, atol=1e-15)


if __name__ == '__main__':
    unittest.main()
