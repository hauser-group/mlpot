import unittest
from mlpot.descriptors import DescriptorSet
import numpy as np


class DescriptorTest():
    class ThreeBodyDescriptorTest(unittest.TestCase):
        cutoff = 8.0

        def cut(self, r):
            return 0.5*(1.0 + np.cos(np.pi*r/self.cutoff))*(r < self.cutoff)

        def dcut(self, r, cutoff=8.0):
            return -0.5*np.sin(
                np.pi*r/self.cutoff)*np.pi/self.cutoff*(r < self.cutoff)

        def test_derivatives_numerically(self):
            with DescriptorSet(['Au', 'Ag'], cutoff=self.cutoff) as ds:
                ds.add_three_body_descriptor('Au', 'Ag', 'Ag', self.name,
                                             self.prms, cuttype='cos')
                types = ['Au', 'Ag', 'Ag', 'Ag', 'Ag', 'Ag']
                xyzs = np.array([[0.0, 0.0, 0.0],
                                 [1.2, 0.0, 0.0],
                                 [-1.1, 0.0, 0.0],
                                 [0.0, 1.2, 0.0],
                                 [1., 2., 3.],
                                 [3, 2., 1.]])

                G = ds.eval_atomwise(types, xyzs)[0][0]
                print(G)
                dG = ds.eval_derivatives_atomwise(types, xyzs)[0][0]
                dG_num = np.zeros_like(dG)
                delta = 1e-5
                for i in range(len(xyzs)):
                    for j in range(3):
                        dx = np.zeros_like(xyzs)
                        dx[i, j] = delta
                        G_plus = ds.eval_atomwise(types, xyzs + dx)[0]
                        G_minus = ds.eval_atomwise(types, xyzs - dx)[0]
                        dG_num[i, j] = (G_plus - G_minus)/(2*delta)
                print(dG)
                print(dG_num)
                np.testing.assert_allclose(dG_num, dG)

        def test_versus_python_implementation(self):
            if callable(self.python_function):
                with DescriptorSet(['Au', 'Ag'], cutoff=self.cutoff) as ds:
                    ds.add_three_body_descriptor('Au', 'Ag', 'Ag', self.name,
                                                 self.prms, cuttype='cos')
                    types = ['Au', 'Ag', 'Ag']
                    rij = .5
                    rik = 1.0
                    costheta = 0
                    sintheta = np.sqrt(1 - costheta**2)
                    print(costheta, sintheta, costheta**2 + sintheta**2)
                    xyzs = np.array([[0.0, 0.0, 0.0],
                                    [rij, 0.0, 0.0],
                                    [rik*costheta, rik*sintheta, 0.0]])
                    print(xyzs)
                    # rij = sqrt((x[1]-x[0])**2)
                    # rik = sqrt((x[2]-x[0])**2 + (y[2]-y[0])**2)
                    # costheta =
                    G = ds.eval_atomwise(types, xyzs)[0]
                    Gp, dGp = self.python_function(rij, rik, costheta)
                    print(dGp)
                    np.testing.assert_allclose(G, Gp)
                    dG = ds.eval_derivatives_atomwise(types, xyzs)[0]
                    print(dG)


class BehlerG4Test(DescriptorTest.ThreeBodyDescriptorTest):
    name = 'BehlerG4auto'
    prms = [1.0, 1.0, 0.4]

    def python_function(self, rij, rik, costheta):
        rjk = np.sqrt(rij**2+rik**2-2*rij*rik*costheta)
        lamb = self.prms[0]
        zeta = self.prms[1]
        eta = self.prms[2]
        exp2_term = 2**(1-zeta)
        cos_term = 1 + lamb*costheta
        exp_term = np.exp(-2*eta*(rij**2+rik**2-rij*rik*costheta))
        return (
            (exp2_term*cos_term**zeta*exp_term
             * self.cut(rij)*self.cut(rik)*self.cut(rjk)),
            [exp2_term*cos_term**(self.prms[1])*exp_term*self.cut(rik)*(
                 self.cut(rjk)*self.dcut(rij)
                 + self.cut(rij)*2*(rik*costheta-2*rij)*eta*self.cut(rjk)
                 + self.cut(rij)*(rik*costheta-rij)*self.dcut(rjk)/rjk),
             2*exp2_term*cos_term**(self.prms[1])*exp_term*(
                 (rik*costheta-2*rij)*self.prms[2]),
             2*exp2_term*cos_term**(self.prms[1]-1)*exp_term*(
                 (rij*costheta-2*rik)*self.prms[2])
             ])


if __name__ == '__main__':
    unittest.main()
