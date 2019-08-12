import unittest
from mlpot.descriptors import DescriptorSet
import numpy as np


class DescriptorTest():
    class DescriptorTest(unittest.TestCase):
        cutoff = 8.0

        def test_derivatives_numerically(self):
            with DescriptorSet(['Au', 'Ag'], cutoff=self.cutoff) as ds:
                self.add_descriptor(ds)
                types = ['Au', 'Ag', 'Ag', 'Ag', 'Ag', 'Ag']
                xyzs = np.array([[0.0, 0.0, 0.0],
                                 [1.2, 0.0, 0.0],
                                 [-1.1, 0.0, 0.0],
                                 [0.0, 1.2, 0.0],
                                 [1., 2., 3.],
                                 [3, 2., 1.]])

                dG = ds.eval_derivatives_atomwise(types, xyzs)[0][0]
                dG_num = np.zeros_like(dG)
                delta = 1e-6
                for i in range(len(xyzs)):
                    for j in range(3):
                        dx = np.zeros_like(xyzs)
                        dx[i, j] = delta
                        G_plus = ds.eval_atomwise(types, xyzs + dx)[0]
                        G_minus = ds.eval_atomwise(types, xyzs - dx)[0]
                        dG_num[i, j] = (G_plus - G_minus)/(2*delta)
                np.testing.assert_allclose(dG_num, dG, atol=1e-9)

    class TwoBodyDescriptorTest(DescriptorTest):
        cutoff = 8.0

        def add_descriptor(self, ds):
            ds.add_two_body_descriptor('Au', 'Ag', self.name,
                                       self.prms, cuttype='cos')

    class ThreeBodyDescriptorTest(DescriptorTest):
        cutoff = 8.0

        def add_descriptor(self, ds):
            ds.add_three_body_descriptor('Au', 'Ag', 'Ag', self.name,
                                         self.prms, cuttype='cos')


class BehlerG1Test(DescriptorTest.TwoBodyDescriptorTest):
    name = 'BehlerG1'
    prms = []


class BehlerG2Test(DescriptorTest.TwoBodyDescriptorTest):
    name = 'BehlerG2'
    prms = [0.4, 2.0]


class BehlerG3Test(DescriptorTest.TwoBodyDescriptorTest):
    name = 'BehlerG3'
    prms = [2.0]


class BehlerG4Test(DescriptorTest.ThreeBodyDescriptorTest):
    name = 'BehlerG4'
    prms = [1.0, 2.0, 0.4]


class BehlerG5Test(DescriptorTest.ThreeBodyDescriptorTest):
    name = 'BehlerG5'
    prms = [1.0, 2.0, 0.4]


if __name__ == '__main__':
    unittest.main()
