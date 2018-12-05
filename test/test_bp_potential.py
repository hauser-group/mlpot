import unittest
import os
from NNpotentials import BPpotential
from NNpotentials.utils import calculate_bp_maps, calculate_bp_indices
import numpy as np
import tensorflow as tf
import pickle

class BPpotentialTest(unittest.TestCase):

    def test_gold_dataset(self):
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

        [Au_atoms], [Au_indices] = calculate_bp_indices(1, Gs_test, types_test)
        test_data= tf.data.Dataset.from_tensor_slices(
            ({'Au_input': np.expand_dims(Au_atoms, axis=0).astype(np.float32),
            'Au_indices': np.expand_dims(Au_indices, axis=0),
            'error_weights': np.expand_dims(
                1.0/np.array(
                    list(map(len, Gs_test)), dtype=np.float32)**2, axis=0)},
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

            # Test feeding the inputs as usual
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

            # Test using the reinitializable iterator
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

    def test_hf_derivatives(self):
        tf.reset_default_graph()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "HF_dataset.pickle"), "rb") as fin:
            try:
                (Gs_test, types_test, E_test,
                dGs_test, F_test) = pickle.load(fin)
            except UnicodeDecodeError as e: # For Python3.6
                (Gs_test, types_test, E_test,
                dGs_test, F_test) = pickle.load(fin, encoding='latin1')
        pot = BPpotential(["H", "F"], [len(Gs_test[0][0]), len(Gs_test[0][1])],
            build_forces = True)

        ([H_atoms, F_atoms], [H_indices, F_indices], [H_derivs, F_derivs]
            ) = calculate_bp_indices(2, Gs_test, types_test, dGs = dGs_test)
        test_data= tf.data.Dataset.from_tensor_slices(
            ({'H_input': H_atoms[np.newaxis, ...].astype(np.float32),
            'H_indices': H_indices[np.newaxis, ...],
            'H_derivatives_input': H_derivs[np.newaxis, ...].astype(np.float32),
            'F_input': F_atoms[np.newaxis, ...].astype(np.float32),
            'F_indices': F_indices[np.newaxis, ...],
            'F_derivatives_input': F_derivs[np.newaxis, ...].astype(np.float32),
            'error_weights': np.expand_dims(
                1.0/np.array(
                    list(map(len, Gs_test)), dtype=np.float32)**2, axis=0)},
            {'energy':np.array(E_test).reshape((1,-1)).astype(np.float32),
             'forces':np.array(F_test)[np.newaxis,...].astype(np.float32)}))

        test_dict = {pot.ANNs['H'].input: H_atoms,
            pot.atom_indices['H']: H_indices,
            pot.ANNs['H'].derivatives_input: H_derivs,
            pot.ANNs['F'].input: F_atoms,
            pot.atom_indices['F']: F_indices,
            pot.ANNs['F'].derivatives_input: F_derivs,
            pot.target:E_test,
            pot.target_forces:F_test,
            pot.rmse_weights: 1.0/np.array(list(map(len, Gs_test)))**2}

        init_op = pot.iterator.make_initializer(test_data)

        with tf.Session() as sess:
            # Not relying on tf.set_seed() as graph level seed depends on
            # the order the graph is build
            np.random.seed(1234)
            for v in pot.variables:
                sess.run(v.assign(np.random.randn(*v.shape)))

            E_control = np.array([5.702612,  5.56348,   5.4710846, 5.4215145,
                      5.4079914, 5.424957,  5.4680867, 5.5339046,
                      5.619496,  5.7223215, 5.840103,  5.970681,
                      6.1118464, 6.261231,  6.416313,  6.5745687,
                      6.7336607, 6.8916163, 7.046926,  7.1985497,
                      7.345853,  7.488504,  7.6263633])
            F_control = np.zeros((23,2,3))
            F_control[:,:,-1] = [[-3.2141814,   3.2141814 ],
                                 [-2.3172617,   2.3172617 ],
                                 [-1.3963969,   1.3963969 ],
                                 [-0.60993123,  0.60993123],
                                 [ 0.05062687, -0.05062687],
                                 [ 0.61372644, -0.61372644],
                                 [ 1.0999537,  -1.0999537 ],
                                 [ 1.5231014,  -1.5231014 ],
                                 [ 1.8921306,  -1.8921306 ],
                                 [ 2.2133608,  -2.2133608 ],
                                 [ 2.4907703,  -2.4907703 ],
                                 [ 2.7250333,  -2.7250333 ],
                                 [ 2.9136925,  -2.9136925 ],
                                 [ 3.0531886,  -3.0531886 ],
                                 [ 3.1416233,  -3.1416233 ],
                                 [ 3.1807523,  -3.1807523 ],
                                 [ 3.17631,    -3.17631   ],
                                 [ 3.136878,   -3.136878  ],
                                 [ 3.0720663,  -3.0720663 ],
                                 [ 2.9908042,  -2.9908042 ],
                                 [ 2.9002442,  -2.9002442 ],
                                 [ 2.805337,   -2.805337  ],
                                 [ 2.7089603,  -2.7089603 ]]

            # Test feeding the inputs as usual
            np.testing.assert_allclose(sess.run(pot.E_predict, test_dict),
                E_control, rtol=1e-5)

            np.testing.assert_array_equal(sess.run(pot.num_atoms, test_dict),
                np.array([2]*23))

            np.testing.assert_allclose(sess.run(pot.F_predict, test_dict),
                F_control, rtol=1e-4)

            # Test using the reinitializable iterator
            sess.run(init_op)
            np.testing.assert_allclose(sess.run(pot.E_predict), E_control,
                rtol=1e-5)

            sess.run(init_op)
            np.testing.assert_array_equal(sess.run(pot.num_atoms),
                np.array([2]*23))

            sess.run(init_op)
            np.testing.assert_allclose(sess.run(pot.F_predict), F_control,
                rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
