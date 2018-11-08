import unittest
import os
from NNpotentials import build_BPestimator
from NNpotentials.utils import calculate_bp_maps, calculate_bp_indices
import numpy as np
import tensorflow as tf
import pickle

class BPpotentialTest(unittest.TestCase):

    def test_gold_dataset(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "Au_BP_testdata.pickle"), "rb") as fin:
            try:
                (Gs_train, types_train, E_train,
                Gs_test, types_test, E_test) = pickle.load(fin)
            except UnicodeDecodeError as e: # For Python3.6
                (Gs_train, types_train, E_train,
                Gs_test, types_test, E_test) = pickle.load(fin, encoding='latin1')
        pot = build_BPestimator(["Au"], [len(Gs_train[0][0])], layers = [[64,64]])

        def train_input_fn():
            batch_size = 50
            def gen():
                for i in range(0, len(Gs_train), batch_size):
                    [Au_atoms], [Au_indices] = calculate_bp_indices(
                        1, Gs_train[i:(i+batch_size)], types_train[i:(i+batch_size)])
                    yield({'Au_input':Au_atoms, 'Au_indices':Au_indices}, E_train[i:(i+batch_size)])
            train_data = tf.data.Dataset.from_generator(gen,
                ({'Au_input':tf.float32, 'Au_indices':tf.int32}, tf.float32),
                ({'Au_input':tf.TensorShape([None, len(Gs_train[0][0])]),
                'Au_indices':tf.TensorShape([None,1])}, tf.TensorShape([None,])))
            return train_data.shuffle(1000).repeat()

        [Au_atoms], [Au_indices] = calculate_bp_indices(1, Gs_test, types_test)

        def test_input_fn():
            test_data= tf.data.Dataset.from_tensor_slices(
                ({'Au_input': np.expand_dims(Au_atoms, axis=0).astype(np.float32),
                  'Au_indices': np.expand_dims(Au_indices, axis=0)},
                  np.array(E_test).reshape((1,-1)).astype(np.float32)))
            return test_data
        pot.train(train_input_fn, steps=100)
        print(pot.evaluate(test_input_fn))
        #with tf.Session() as sess:
            # Not relying on tf.set_seed() as graph level seed depends on
            # the order the graph is build
            #np.random.seed(1234)
            #for v in pot.variables:
            #    sess.run(v.assign(np.random.randn(*v.shape)))

            #np.testing.assert_allclose(sess.run(pot.E_predict, test_dict),
            #    np.array([-13.735861,   -9.856386,   -8.934874,  -13.685179,
            #              -13.685591,  -12.313505,  -12.989342,  -13.678537,
            #              -12.663105,  -13.094957,  -10.074066,   -7.7194157,
            #              -13.338873,   -8.050451,   -7.3590875, -11.71219,
            #              -10.556736,  -17.370564,  -13.613234,  -13.5924,
            #              -12.43917,   -13.568087,   -7.9591656, -12.175657,
            #              -13.432264,  -19.11342,   -13.68409,   -12.032116,
            #              -11.541302,   -8.347027,  -7.5450783]), rtol=1e-5)

            #np.testing.assert_array_equal(sess.run(pot.num_atoms, test_dict),
            #    np.array([2]*31))


if __name__ == '__main__':
    unittest.main()
