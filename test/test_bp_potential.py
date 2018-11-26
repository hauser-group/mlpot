import unittest
import os
from NNpotentials import BPpotential
from NNpotentials.utils import calculate_bp_maps, calculate_bp_indices
import numpy as np
import tensorflow as tf
import pickle

class BPpotentialTest(unittest.TestCase):

    def test_gold_dataset_placeholder(self):
        tf.reset_default_graph()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "Au_BP_testdata.pickle"), "rb") as fin:
            try:
                (Gs_train, types_train, E_train,
                Gs_test, types_test, E_test) = pickle.load(fin)
            except UnicodeDecodeError as e: # For Python3.6
                (Gs_train, types_train, E_train,
                Gs_test, types_test, E_test) = pickle.load(fin, encoding='latin1')
        pot = BPpotential(["Au"], [len(Gs_train[0][0])], layers = [[64,64]])

        #[Au_atoms], [Au_maps] = calculate_bp_maps(1, Gs_test, types_test)
        #test_dict = {pot.ANNs["Au"].input: Au_atoms,
        #    pot.atom_maps["Au"]: Au_maps,
        #    pot.target:E_test, pot.rmse_weights: 1.0/np.array(list(map(len, Gs_test)))**2}
        [Au_atoms], [Au_indices] = calculate_bp_indices(1, Gs_test, types_test)
        test_dict = {pot.ANNs["Au"].input: Au_atoms,
            pot.atom_indices["Au"]: Au_indices,
            pot.target:E_test, pot.rmse_weights: 1.0/np.array(list(map(len, Gs_test)))**2}

        with tf.Session() as sess:
            # Not relying on tf.set_seed() as graph level seed depends on
            # the order the graph is build
            np.random.seed(1234)
            for v in pot.variables:
                sess.run(v.assign(np.random.randn(*v.shape)))

            np.testing.assert_allclose(sess.run(pot.E_predict, test_dict),
                np.array([-13.735861,   -9.856386,   -8.934874,  -13.685179,
                          -13.685591,  -12.313505,  -12.989342,  -13.678537,
                          -12.663105,  -13.094957,  -10.074066,   -7.7194157,
                          -13.338873,   -8.050451,   -7.3590875, -11.71219,
                          -10.556736,  -17.370564,  -13.613234,  -13.5924,
                          -12.43917,   -13.568087,   -7.9591656, -12.175657,
                          -13.432264,  -19.11342,   -13.68409,   -12.032116,
                          -11.541302,   -8.347027,  -7.5450783]), rtol=1e-5)

            np.testing.assert_array_equal(sess.run(pot.num_atoms, test_dict),
                np.array([2]*31))

    def test_gold_dataset_iterator(self):
        tf.reset_default_graph()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "Au_BP_testdata.pickle"), "rb") as fin:
            try:
                (Gs_train, types_train, E_train,
                Gs_test, types_test, E_test) = pickle.load(fin)
            except UnicodeDecodeError as e: # For Python3.6
                (Gs_train, types_train, E_train,
                Gs_test, types_test, E_test) = pickle.load(fin, encoding='latin1')
        pot = BPpotential(["Au"], [len(Gs_train[0][0])],
            input_mode = 'iterator', layers = [[64,64]])

        #[Au_atoms], [Au_maps] = calculate_bp_maps(1, Gs_test, types_test)
        #test_dict = {pot.ANNs["Au"].input: Au_atoms,
        #    pot.atom_maps["Au"]: Au_maps,
        #    pot.target:E_test, pot.rmse_weights: 1.0/np.array(list(map(len, Gs_test)))**2}
        [Au_atoms], [Au_indices] = calculate_bp_indices(1, Gs_test, types_test)
        test_data= tf.data.Dataset.from_tensor_slices(
            ({'Au_input': np.expand_dims(Au_atoms, axis=0).astype(np.float32),
            'Au_indices': np.expand_dims(Au_indices, axis=0),
            'error_weights': np.expand_dims(
                1.0/np.array(map(len, Gs_test)).astype(np.float32)**2, axis=0)},
            {'energy':np.array(E_test).reshape((1,-1)).astype(np.float32)}))

        test_dict = {pot.ANNs["Au"].input: Au_atoms,
            pot.atom_indices["Au"]: Au_indices,
            pot.target:E_test,
            pot.rmse_weights: 1.0/np.array(list(map(len, Gs_test)))**2}

        init_op = pot.iterator.make_initializer(test_data)

        with tf.Session() as sess:
            # Not relying on tf.set_seed() as graph level seed depends on
            # the order the graph is build
            np.random.seed(1234)
            for v in pot.variables:
                sess.run(v.assign(np.random.randn(*v.shape)))

            np.testing.assert_allclose(sess.run(pot.E_predict, test_dict),
                np.array([-13.735861,   -9.856386,   -8.934874,  -13.685179,
                          -13.685591,  -12.313505,  -12.989342,  -13.678537,
                          -12.663105,  -13.094957,  -10.074066,   -7.7194157,
                          -13.338873,   -8.050451,   -7.3590875, -11.71219,
                          -10.556736,  -17.370564,  -13.613234,  -13.5924,
                          -12.43917,   -13.568087,   -7.9591656, -12.175657,
                          -13.432264,  -19.11342,   -13.68409,   -12.032116,
                          -11.541302,   -8.347027,  -7.5450783]), rtol=1e-5)

            np.testing.assert_array_equal(sess.run(pot.num_atoms, test_dict),
                np.array([2]*31))

            sess.run(init_op)
            np.testing.assert_allclose(sess.run(pot.E_predict),
                np.array([-13.735861,   -9.856386,   -8.934874,  -13.685179,
                          -13.685591,  -12.313505,  -12.989342,  -13.678537,
                          -12.663105,  -13.094957,  -10.074066,   -7.7194157,
                          -13.338873,   -8.050451,   -7.3590875, -11.71219,
                          -10.556736,  -17.370564,  -13.613234,  -13.5924,
                          -12.43917,   -13.568087,   -7.9591656, -12.175657,
                          -13.432264,  -19.11342,   -13.68409,   -12.032116,
                          -11.541302,   -8.347027,  -7.5450783]), rtol=1e-5)

            sess.run(init_op)
            np.testing.assert_array_equal(sess.run(pot.num_atoms),
                np.array([2]*31))


if __name__ == '__main__':
    unittest.main()
