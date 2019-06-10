import unittest
import os
from NNpotentials import (SMATBpotential, NNEpotential, NNRHOpotential,
    NNERHOpotential, NNVERHOpotential, NNfreeERHOpotential)
from NNpotentials.utils import calculate_eam_maps
import numpy as np
import tensorflow as tf
import pickle

class EAMpotentialTest(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "Au_EAM_testdata.pickle"), "rb") as fin:
            try:
                (self.Gs_train, self.types_train, self.E_train,
                self.Gs_test, self.types_test, self.E_test) = pickle.load(fin)
            except UnicodeDecodeError as e: # For Python3.6
                (self.Gs_train, self.types_train, self.E_train,
                self.Gs_test, self.types_test, self.E_test) = pickle.load(
                    fin, encoding='latin1')

        ([[self.r_Au_Au]], [[self.b_map_Au_Au]],
            [self.Au_map]) = calculate_eam_maps(
            1, self.Gs_test, self.types_test)

        self.initial_params = {}
        self.initial_params[("A", "Au", "Au")] = 0.2061
        self.initial_params[("xi", "Au", "Au")] = 1.7900
        self.initial_params[("p", "Au", "Au")] = 10.229
        self.initial_params[("q", "Au", "Au")] = 4.036
        self.initial_params[("r0", "Au", "Au")] = 2.88

    def tearDown(self):
        tf.reset_default_graph()


class SMATBpotentialTest(EAMpotentialTest):

    def test_gold_dataset(self):
        pot = SMATBpotential(["Au"], initial_params = self.initial_params)

        test_dict = {pot.ANNs["Au"].inputs["Au"]: self.r_Au_Au,
             pot.ANNs["Au"].b_maps["Au"]: self.b_map_Au_Au,
             pot.atom_maps["Au"]: self.Au_map,
             pot.target: self.E_test,
             pot.rmse_weights: 1.0/np.array(list(map(len, self.Gs_test)))**2}

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(pot.variables))

            np.testing.assert_allclose(
                sess.run(pot.E_predict, test_dict),
                np.array([-4.94568,     -2.6547625,  -0.6748188,   -0.013763459,
                          -0.012277357, -0.16663098, -4.6700187,   -0.019066766,
                          -0.13481946,  -0.09345694, -3.4406862,   -1.5484986,
                          -0.06844438,  -0.8315902,  -1.349937,    -0.22047868,
                          -0.35986638,  -6.9231143,  -0.034387566, -0.037684362,
                          -0.15533087,  -0.04122702, -1.6580167,   -4.4032536,
                          -0.058135785, -7.503875,   -0.015384224, -0.191679,
                          -0.23645946,  -0.77570075, -0.95553195 ]), rtol=1e-5)

class NNEpotentialTest(EAMpotentialTest):

    def test_gold_dataset(self):
        pot = NNEpotential(["Au"], initial_params = self.initial_params)

        test_dict = {pot.ANNs["Au"].inputs["Au"]: self.r_Au_Au,
             pot.ANNs["Au"].b_maps["Au"]: self.b_map_Au_Au,
             pot.atom_maps["Au"]: self.Au_map,
             pot.target: self.E_test,
             pot.rmse_weights: 1.0/np.array(list(map(len, self.Gs_test)))**2}

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(pot.variables))
            # Not relying on tf.set_seed() as graph level seed depends on
            # the order the graph is build
            np.random.seed(1234)
            for v in pot.variables:
                sess.run(v.assign(np.random.randn(*v.shape)))

            np.testing.assert_allclose(
                sess.run(pot.E_predict, test_dict),
                np.array([ 23.52703,   50.121662, 117.75035,  132.14636,
                          127.72339,   99.749146,  25.486397, 142.7019,
                          110.48804,  128.13019,   38.30233,   75.23668,
                          140.15025,  109.039215,  82.706024,  86.36736,
                          124.003815,  12.756134, 151.26387,  151.13742,
                          103.28669,  150.60617,   71.71126,   27.66743,
                          144.85855,    8.327611, 136.125,     92.89981,
                           83.19737,  112.22593, 102.01085 ]), rtol=1e-5)

class NNRHOpotentialTest(EAMpotentialTest):

    def test_gold_dataset(self):
        pot = NNRHOpotential(["Au"], initial_params = self.initial_params)

        test_dict = {pot.ANNs["Au"].inputs["Au"]: self.r_Au_Au,
             pot.ANNs["Au"].b_maps["Au"]: self.b_map_Au_Au,
             pot.atom_maps["Au"]: self.Au_map,
             pot.target: self.E_test,
             pot.rmse_weights: 1.0/np.array(list(map(len, self.Gs_test)))**2}

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(pot.variables))
            # Not relying on tf.set_seed() as graph level seed depends on
            # the order the graph is build
            np.random.seed(1234)
            for v in pot.variables:
                sess.run(v.assign(np.random.randn(*v.shape)))

            np.testing.assert_allclose(
                sess.run(pot.E_predict, test_dict),
                np.array([10.852601,  22.81675,    63.20938,  68.71097,
                          65.520744,  51.087536,  11.810835,  76.55871,
                          57.93685,   69.451355,  17.343874,  37.466385,
                          77.292854,  58.018543,  42.035255,  42.93869,
                          65.675865,   5.2400026, 83.7398,    83.84186,
                          53.32183,   83.65372,   35.30896,   12.815254,
                          80.292946,   3.3003106, 71.62463,   46.846622,
                          41.102154,  59.93209,   53.77338  ]), rtol=1e-5)

class NNERHOpotentialTest(EAMpotentialTest):

    def test_gold_dataset(self):
        pot = NNERHOpotential(["Au"], initial_params = self.initial_params)

        test_dict = {pot.ANNs["Au"].inputs["Au"]: self.r_Au_Au,
             pot.ANNs["Au"].b_maps["Au"]: self.b_map_Au_Au,
             pot.atom_maps["Au"]: self.Au_map,
             pot.target: self.E_test,
             pot.rmse_weights: 1.0/np.array(list(map(len, self.Gs_test)))**2}

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(pot.variables))
            # Not relying on tf.set_seed() as graph level seed depends on
            # the order the graph is build
            np.random.seed(1234)
            for v in pot.variables:
                sess.run(v.assign(np.random.randn(*v.shape)))

            np.testing.assert_allclose(
                sess.run(pot.E_predict, test_dict),
                np.array([ 76.809944,  78.04877,  101.2234,   93.335205,
                           89.42229,   82.6702,    76.97342,  103.12275,
                           89.74382,  101.379395,  77.49617,   83.39936,
                          108.90043,   97.54176,   86.18508,   74.254715,
                           99.88108,   75.17703,  113.24658,  113.72391,
                           84.98351,  113.878525,  82.23049,   77.11247,
                          111.56587,   74.350746,  96.93275,   78.28093,
                           72.37287,   98.932724,  94.420425]), rtol=1e-5)

class NNVERHOpotentialTest(EAMpotentialTest):

    def test_gold_dataset(self):
        pot = NNVERHOpotential(["Au"], initial_params = self.initial_params)

        test_dict = {pot.ANNs["Au"].inputs["Au"]: self.r_Au_Au,
             pot.ANNs["Au"].b_maps["Au"]: self.b_map_Au_Au,
             pot.atom_maps["Au"]: self.Au_map,
             pot.target: self.E_test,
             pot.rmse_weights: 1.0/np.array(list(map(len, self.Gs_test)))**2}

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(pot.variables))
            # Not relying on tf.set_seed() as graph level seed depends on
            # the order the graph is build
            np.random.seed(1234)
            for v in pot.variables:
                sess.run(v.assign(np.random.randn(*v.shape)))

            np.testing.assert_allclose(
                sess.run(pot.E_predict, test_dict),
                np.array([15.349937,  16.332008,  16.819798,   5.3563538,
                           5.138779,  12.319517,  15.48003,    6.4758625,
                          12.386071,  12.169296,  16.012493,  16.793566,
                          11.557325,  16.94801,   16.878399,  12.167745,
                          15.409866,  14.116678,   9.069218,   9.457203,
                          12.351418,   9.826996,  16.745457,  15.601319,
                          11.0845785, 13.35572,    5.669809,  12.245018,
                          12.128244,  16.918398,  16.973438 ]), rtol=1e-5)

class NNfreeERHOpotentialTest(EAMpotentialTest):

    def test_gold_dataset(self):
        pot = NNfreeERHOpotential(["Au"], initial_params = self.initial_params)

        test_dict = {pot.ANNs["Au"].inputs["Au"]: self.r_Au_Au,
             pot.ANNs["Au"].b_maps["Au"]: self.b_map_Au_Au,
             pot.atom_maps["Au"]: self.Au_map,
             pot.target: self.E_test,
             pot.rmse_weights: 1.0/np.array(list(map(len, self.Gs_test)))**2}

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(pot.variables))
            # Not relying on tf.set_seed() as graph level seed depends on
            # the order the graph is build
            np.random.seed(1234)
            for v in pot.variables:
                sess.run(v.assign(np.random.randn(*v.shape)))

            np.testing.assert_allclose(
                sess.run(pot.E_predict, test_dict),
                np.array([ 8.550756, 19.277721, 58.195335, 63.86586,
                          60.793076, 45.632095,  9.384782, 71.4253,
                          52.478893, 63.988274, 14.290964, 33.133484,
                          71.82463,  53.142082, 37.555336, 37.491238,
                          60.323048,  3.799034, 78.32719,  78.41045,
                          47.8654,   78.208466, 31.05736,  10.264334,
                          74.825134,  2.238496, 66.6718,   41.39421,
                          35.65802,  55.008442, 48.99653 ]), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
