import unittest
import os
from NNpotentials import BPpotential
from NNpotentials.utils import calculate_bp_maps
import numpy as np
import tensorflow as tf
import pickle

class BPpotentialTest(unittest.TestCase):

    def test_gold_dataset(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "Au_BP_testdata.pickle"), "rb") as fin:
            (Gs_train, types_train, E_train,
            Gs_test, types_test, E_test) = pickle.load(fin)

        pot = BPpotential(["Au"], [len(Gs_train[0][0])], layers = [[64,64]])

        [Au_atoms], [Au_maps] = calculate_bp_maps(1, Gs_test, types_test)
        test_dict = {pot.ANNs["Au"].input: Au_atoms,
            pot.atom_maps["Au"]: Au_maps,
            pot.target:E_test, pot.rmse_weights: 1.0/np.array(map(len, Gs_test))**2}

        with tf.Session() as sess:
            # Not relying on tf.set_seed() as graph level seed depends on
            # the order the graph is build
            np.random.seed(1234)
            for v in pot.variables:
                sess.run(v.assign(np.random.randn(*v.shape)))

            np.testing.assert_array_almost_equal(sess.run(pot.E_predict, test_dict),
                np.array([-13.735861,   -9.856386,   -8.934874,  -13.685179,
                          -13.685591,  -12.313505,  -12.989342,  -13.678537,
                          -12.663105,  -13.094957,  -10.074066,   -7.7194157,
                          -13.338873,   -8.050451,   -7.3590875, -11.71219,
                          -10.556736,  -17.370564,  -13.613234,  -13.5924,
                          -12.43917,   -13.568087,   -7.9591656, -12.175657,
                          -13.432264,  -19.11342,   -13.68409,   -12.032116,
                          -11.541302,   -8.347027,  -7.5450783]), decimal=5)


if __name__ == '__main__':
    unittest.main()
