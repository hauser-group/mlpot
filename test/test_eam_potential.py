import unittest
import os
from NNpotentials import SMATBpotential
from NNpotentials.utils import calculate_eam_maps
import numpy as np
import tensorflow as tf
import pickle

class BPpotentialTest(unittest.TestCase):

    def test_gold_dataset(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "Au_EAM_testdata.pickle"), "rb") as fin:
            (Gs_train, types_train, E_train,
            Gs_test, types_test, E_test) = pickle.load(fin)

        initial_params = {}
        initial_params[("A", "Au", "Au")] = 0.2061
        initial_params[("xi", "Au", "Au")] = 1.7900
        initial_params[("p", "Au", "Au")] = 10.229
        initial_params[("q", "Au", "Au")] = 4.036
        initial_params[("r0", "Au", "Au")] = 2.88

        pot = SMATBpotential(["Au"], initial_params = initial_params)

        ([[r_Au_Au]], [[b_map_Au_Au]], [Au_map]) = calculate_eam_maps(
            1, Gs_test, types_test)
        test_dict = {pot.ANNs["Au"].inputs["Au"]: r_Au_Au,
             pot.ANNs["Au"].b_maps["Au"]: b_map_Au_Au,
             pot.atom_maps["Au"]: Au_map,
             pot.target: E_test,
             pot.rmse_weights: 1.0/np.array(map(len, Gs_test))**2}

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(pot.variables))

            np.testing.assert_array_almost_equal(sess.run(pot.E_predict, test_dict),
                np.array([-4.94568,     -2.6547625,   -0.6748188,   -0.013763459,
                          -0.012277357, -0.16663098,  -4.6700187,   -0.019066766,
                          -0.13481946,  -0.09345694,  -3.4406862,   -1.5484986,
                          -0.06844438,  -0.8315902,   -1.349937,    -0.22047868,
                          -0.35986638,  -6.9231143,   -0.034387566, -0.037684362,
                          -0.15533087,  -0.04122702,  -1.6580167,   -4.4032536,
                          -0.058135785, -7.503875,    -0.015384224, -0.191679,
                          -0.23645946,  -0.77570075,  -0.95553195 ]), decimal=5)


if __name__ == '__main__':
    unittest.main()
