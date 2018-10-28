import unittest
from NNpotentials import BPpotential
from NNpotentials.utils import calculate_bp_maps
import numpy as np
import tensorflow as tf
import cPickle

class BPpotentialTest(unittest.TestCase):

    def test_gold_dataset(self):
        with open("Au_testdata.pickle", "rb") as fin:
            (Gs_train, types_train, E_train, 
            Gs_test, types_test, E_test) = cPickle.load(fin)

        pot = BPpotential(["Au"], [len(Gs_train[0][0])], layers = [[64,64]])

        [Au_atoms], [Au_maps] = calculate_bp_maps(1, Gs_test, types_test)
        test_dict = {pot.ANNs["Au"].input: Au_atoms, 
            pot.atom_maps["Au"]: Au_maps, 
            pot.target:E_test, pot.rmse_weights: 1.0/np.array(map(len, Gs_test))**2}
        
        with tf.Session() as sess:
            np.random.seed(1234)
            for v in pot.variables:
                sess.run(v.assign(np.random.randn(*v.shape)))
             
            np.testing.assert_array_almost_equal(sess.run(pot.E_predict, test_dict),
                np.array([-13.735861,   -9.856385,   -8.934873,  -13.68518,
                          -13.685591,  -12.313506,  -12.989344,  -13.678535,
                          -12.663107,  -13.094959,  -10.074065,   -7.7194138,
                          -13.338874,   -8.050451,   -7.3590865, -11.712188,
                          -10.556735,  -17.370564,   -13.613236,  -13.592398,
                          -12.439171,  -13.568089,   -7.9591637, -12.175658,
                          -13.432264,  -19.11342,   -13.68409,   -12.032118,
                          -11.541304,   -8.347028,   -7.5450773]))


if __name__ == '__main__':
    unittest.main()
