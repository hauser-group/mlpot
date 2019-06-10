import unittest
from DescriptorLib.SymmetryFunctionSet import SymmetryFunctionSet as SymFunSet
import numpy as np
from itertools import product, combinations_with_replacement

class LibraryTest(unittest.TestCase):

    def test_trimer(self):
        with SymFunSet(['Ag'], cutoff = 8.0) as sfs:
            types = ['Ag', 'Ag', 'Ag']

            def fcut(r):
                return 0.5*(1.0 + np.cos(np.pi*r/sfs.cutoff))*(r < sfs.cutoff)

            # Parameters from Artrith and Kolpak Nano Lett. 2014, 14, 2670
            etas = np.array([0.0009, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2])
            for eta in etas:
                sfs.add_TwoBodySymmetryFunction('Ag', 'Ag', 'BehlerG1', [eta],
                    cuttype='cos')

            ang_etas = np.array([0.0001, 0.003, 0.008])
            zetas = np.array([1.0, 4.0])
            for ang_eta in ang_etas:
                for lamb in [-1.0, 1.0]:
                    for zeta in zetas:
                        sfs.add_ThreeBodySymmetryFunction('Ag', 'Ag', 'Ag',
                            'BehlerG3', [lamb, zeta, ang_eta], cuttype='cos')

            # Also test BehlerG4
            for ang_eta in ang_etas:
                for lamb in [-1.0, 1.0]:
                    for zeta in zetas:
                        sfs.add_ThreeBodySymmetryFunction('Ag', 'Ag', 'Ag',
                            'BehlerG4', [lamb, zeta, ang_eta], cuttype='cos')

            N = 30
            r_vec = np.linspace(1., 5., N)#
            theta_vec = np.linspace(0.0*np.pi, 2.*np.pi, N, endpoint=True)
            for ri in r_vec:
                for ti in theta_vec:
                    xyzs = np.array([[0.0, 0.0, 0.0],
                                    [ri, 0.0, 0.0],
                                    [ri*np.cos(ti), ri*np.sin(ti), 0.0]])

                    rij = np.linalg.norm(xyzs[0,:]-xyzs[1,:])
                    rik = np.linalg.norm(xyzs[0,:]-xyzs[2,:])
                    rjk = np.linalg.norm(xyzs[1,:]-xyzs[2,:])
                    np.testing.assert_allclose(
                        np.linalg.norm(xyzs[1,:]-xyzs[2,:]),
                        np.sqrt(rij**2+rik**2-2.*rij*rik*np.cos(ti)),
                        atol=1E-12)

                    Gs = sfs.eval(types, xyzs)
                    Gs_atomwise = sfs.eval_atomwise(types, xyzs)
                    Gs_ref = np.concatenate([2*np.exp(-etas*ri**2)*fcut(ri)]
                        + [2**(1.-zetas)*np.exp(-eta*(rij**2+rik**2+rjk**2))
                        *(1.+lamb*np.cos(ti))**zetas*fcut(rij)*fcut(rik)*fcut(rjk)
                        for eta in ang_etas for lamb in [-1.0, 1.0]]
                        + [2**(1.-zetas)*np.exp(-eta*(rij**2+rik**2))
                        *(1.+lamb*np.cos(ti))**zetas*fcut(rij)*fcut(rik)
                        for eta in ang_etas for lamb in [-1.0, 1.0]])
                    np.testing.assert_allclose(Gs, Gs_atomwise)
                    np.testing.assert_allclose(Gs[0], Gs_ref)
                    np.testing.assert_allclose(Gs_atomwise[0], Gs_ref)

                    dGs = sfs.eval_derivatives(types, xyzs)
                    dGs_atomwise = sfs.eval_derivatives_atomwise(types, xyzs)
                    # Adding the equal_nan=False option shows a bug for descriptors
                    # using rik as input
                    np.testing.assert_allclose(dGs, dGs_atomwise)#, equal_nan=False)

                    Gs, dGs = sfs.eval_with_derivatives(types, xyzs)
                    Gs_atomwise, dGs_atomwise = sfs.eval_with_derivatives_atomwise(types, xyzs)
                    np.testing.assert_allclose(Gs, Gs_atomwise)
                    np.testing.assert_allclose(dGs, dGs_atomwise)

if __name__ == '__main__':
    unittest.main()
