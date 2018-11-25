import tensorflow as _tf
import numpy as _np
import abc

precision = _tf.float32

def nn_layer(input_tensor, input_dim, output_dim, act = _tf.nn.tanh,
  initial_bias = None, name = "layer"):
    with _tf.variable_scope(name):
        weights = _tf.get_variable("w", dtype = precision,
            shape = [input_dim, output_dim], initializer = _tf.random_normal_initializer(
            stddev = 1./_np.sqrt(input_dim), dtype = precision),
            collections = [_tf.GraphKeys.MODEL_VARIABLES,
                            _tf.GraphKeys.REGULARIZATION_LOSSES,
                            _tf.GraphKeys.GLOBAL_VARIABLES])
        if initial_bias == None:
            biases = _tf.get_variable("b", dtype = precision, shape = [output_dim],
                initializer = _tf.constant_initializer(0.0, dtype = precision),
                collections = [_tf.GraphKeys.MODEL_VARIABLES,
                            _tf.GraphKeys.GLOBAL_VARIABLES])
        else:
            biases = _tf.get_variable("b", dtype = precision, shape = [output_dim],
                initializer = _tf.constant_initializer(initial_bias, dtype = precision),
                collections = [_tf.GraphKeys.MODEL_VARIABLES,
                            _tf.GraphKeys.GLOBAL_VARIABLES])
        preactivate = _tf.nn.xw_plus_b(input_tensor, weights, biases)
        if act == None:
            activations = preactivate
        else:
            activations = act(preactivate)
        _tf.summary.histogram("weights", weights)
        _tf.summary.histogram("biases", biases)
        _tf.summary.histogram("activations", activations)
        return activations, weights, biases

class AtomicEnergyPotential(object):

    def __init__(self, atom_types, **kwargs):
        #self.target = _tf.placeholder(shape = (None,), dtype = precision,
        #    name = "target")
        self.atom_types = atom_types
        self.error_scaling = kwargs.get("error_scaling", 1000)

        self.atomic_contributions = {}
        self.atom_maps = {}
        self.atom_indices = {}

        #for t in self.atom_types:
        #    self.atom_indices[t] = _tf.placeholder(shape = (None,1),
        #        dtype = _tf.int32, name = "{}_indices".format(t))
        #    self.atom_maps[t] = _tf.sparse_placeholder(shape = (None, None),
        #        dtype = precision, name = "{}_map".format(t))

        self.configureAtomicContributions(**kwargs)
        # Convenience handle
        self.ANNs = self.atomic_contributions
        self.target = self.labels['energy']
        for t in self.atom_types:
            self.atom_indices[t] = self.features['%s_indices'%t]

        self.E_predict = _tf.scatter_nd(
            _tf.concat([self.atom_indices[t] for t in self.atom_types], 0),
            _tf.concat([_tf.reshape(self.atomic_contributions[t].output, [-1])
            for t in self.atom_types], 0), _tf.shape(self.target),
            name = "E_prediction")
        #self.E_predict = _tf.reduce_sum([
        #    _tf.sparse_tensor_dense_matmul(self.atom_maps[t],
        #    self.atomic_contributions[t].output) for t in self.atom_types],
        #    axis = [0, 2], name = "E_prediction")

        #self.num_atoms =  _tf.reduce_sum(
        #    [_tf.sparse_reduce_sum(self.atom_maps[t], axis = 1) for t in self.atom_types],
        #    axis = 0, name = "NumberOfAtoms")
        self.num_atoms = _tf.reduce_sum([_tf.bincount(self.atom_indices[t])
            for t in self.atom_types], axis = 0, name = "NumberOfAtoms")
        # Tensorflow operation that calculates the sum squared error per atom.
        # Note that the whole error per atom is squared.
        with _tf.name_scope("RMSE"):
            self.rmse_weights = _tf.placeholder(shape = (None,),
                dtype = precision, name = "weights")
            self.rmse = self.error_scaling*_tf.sqrt(_tf.reduce_mean(
                (self.target-self.E_predict)**2*self.rmse_weights))
            #self.rmse = self.error_scaling*_tf.sqrt(
            #    _tf.losses.mean_squared_error(self.target,
            #    self.E_predict, weights = 1.0/self.num_atoms**2))
            self.rmse_summ = _tf.summary.scalar("RMSE", self.rmse, family = "performance")

        self.variables = _tf.get_collection(_tf.GraphKeys.MODEL_VARIABLES,
            scope = _tf.get_default_graph().get_name_scope())
        self.saver = _tf.train.Saver(self.variables, max_to_keep = None, save_relative_paths = True)

    @abc.abstractmethod
    def configureAtomicContributions(self):
        pass
