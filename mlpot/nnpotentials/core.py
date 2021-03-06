import tensorflow.compat.v1 as _tf
import numpy as _np
import abc
_tf.disable_v2_behavior()


def nn_layer(input_tensor, input_dim, output_dim, act=_tf.nn.tanh,
             initial_bias=None, name='layer', precision=_tf.float32):
    with _tf.variable_scope(name):
        weights = _tf.get_variable(
            'w', dtype=precision, shape=[input_dim, output_dim],
            initializer=_tf.random_normal_initializer(
                stddev=1./_np.sqrt(input_dim), dtype=precision),
            collections=[_tf.GraphKeys.MODEL_VARIABLES,
                         _tf.GraphKeys.REGULARIZATION_LOSSES,
                         _tf.GraphKeys.GLOBAL_VARIABLES])
        if initial_bias is None:
            biases = _tf.get_variable(
                'b', dtype=precision, shape=[output_dim],
                initializer=_tf.constant_initializer(0.0, dtype=precision),
                collections=[_tf.GraphKeys.MODEL_VARIABLES,
                             _tf.GraphKeys.GLOBAL_VARIABLES])
        else:
            biases = _tf.get_variable(
                'b', dtype=precision, shape=[output_dim],
                initializer=_tf.constant_initializer(
                    initial_bias, dtype=precision),
                collections=[_tf.GraphKeys.MODEL_VARIABLES,
                             _tf.GraphKeys.GLOBAL_VARIABLES])
        preactivate = _tf.nn.xw_plus_b(input_tensor, weights, biases)
        if act is None:
            activations = preactivate
        else:
            activations = act(preactivate)
        _tf.summary.histogram('weights', weights)
        _tf.summary.histogram('biases', biases)
        _tf.summary.histogram('activations', activations)
        return activations, weights, biases


class AtomicEnergyPotential(object):

    def __init__(self, atom_types, **kwargs):
        self.atom_types = atom_types
        self.build_forces = kwargs.get('build_forces', False)
        self.precision = kwargs.get('precision', _tf.float32)

        self.atomic_contributions = {}
        self.atom_indices = {}

        self.feature_types = {'error_weights': self.precision}
        self.feature_shapes = {'error_weights': _tf.TensorShape([None, ])}

        for t in self.atom_types:
            self.feature_types['%s_indices' % t] = _tf.int32
            self.feature_shapes['%s_indices' % t] = _tf.TensorShape([None, 1])

        self.label_types = {'energy': self.precision}
        self.label_shapes = {'energy': _tf.TensorShape([None, ])}
        if self.build_forces:
            self.label_types['forces'] = self.precision
            self.label_shapes['forces'] = _tf.TensorShape([None, None, 3])

        # The individual atomic contributions and the data iterator is set up
        # in this abstact method that has to be implemented for each potential
        # type
        self.configureAtomicContributions(**kwargs)

        # Convenience handle for backwards compatibility
        self.ANNs = self.atomic_contributions

        self.target = self.labels['energy']
        if self.build_forces:
            self.target_forces = self.labels['forces']
        for t in self.atom_types:
            self.atom_indices[t] = self.features['%s_indices' % t]

        with _tf.name_scope('Energy_Calculation'):
            self.E_predict = _tf.scatter_nd(
                _tf.concat([self.atom_indices[t] for t in self.atom_types], 0),
                _tf.concat([
                    _tf.reshape(self.atomic_contributions[t].output, [-1])
                    for t in self.atom_types], 0), _tf.shape(self.target),
                name='E_prediction')

        with _tf.name_scope('Atom_Counting'):
            self.num_atoms = _tf.reduce_sum([_tf.bincount(self.atom_indices[t])
                                             for t in self.atom_types],
                                            axis=0, name='NumberOfAtoms')

        with _tf.name_scope('MSE'):
            self.error_weights = self.features['error_weights']
            self.mse = _tf.reduce_mean(
                (self.target-self.E_predict)**2*self.error_weights)
            self.mse_summ = _tf.summary.scalar(
                'MSE', self.mse, family='performance')

        with _tf.name_scope('RMSE'):
            self.error_weights = self.features['error_weights']
            self.rmse = _tf.sqrt(_tf.reduce_mean(
                (self.target-self.E_predict)**2*self.error_weights))
            self.rmse_summ = _tf.summary.scalar(
                'RMSE', self.rmse, family='performance')

        if self.build_forces:
            with _tf.name_scope('Force_Calculation'):
                self.gradients = {}
                for t in self.atom_types:
                    self.gradients[t] = _tf.gradients(
                        self.atomic_contributions[t].output,
                        self.atomic_contributions[t].input,
                        name='%s_gradients' % t)[0]

                self.F_predict = _tf.negative(_tf.scatter_nd(
                    _tf.concat([
                        self.atom_indices[t] for t in self.atom_types], 0),
                    _tf.concat([_tf.einsum(
                        'ijkl,ij->ikl',
                        self.atomic_contributions[t].derivatives_input,
                        self.gradients[t], name='%s_einsum' % t)
                        for t in self.atom_types], 0),
                    _tf.shape(self.target_forces), name='F_prediction'))

            with _tf.name_scope('MSE_Forces'):
                # Following Behler. Int. J. Quant. Chem. 2015 115, 1032-1050
                # equation (21). !! Not the same !! for varying number of atoms
                self.mse_forces = _tf.reduce_mean(
                    _tf.reduce_mean((self.target_forces-self.F_predict)**2,
                                    axis=[1, 2])*self.error_weights)
                self.mse_forces_summ = _tf.summary.scalar(
                    'MSE_Forces', self.mse_forces, family='performance')

            with _tf.name_scope('RMSE_Forces'):
                # Following Behler. Int. J. Quant. Chem. 2015 115, 1032-1050
                # equation (21). !! Not the same !! for varying number of atoms
                self.rmse_forces = _tf.sqrt(_tf.reduce_mean(
                    _tf.reduce_mean((self.target_forces-self.F_predict)**2,
                                    axis=[1, 2])*self.error_weights))
                self.rmse_forces_summ = _tf.summary.scalar(
                    'RMSE_Forces', self.rmse_forces, family='performance')

        self.variables = _tf.get_collection(
            _tf.GraphKeys.MODEL_VARIABLES,
            scope=_tf.get_default_graph().get_name_scope())
        self.saver = _tf.train.Saver(
            self.variables, max_to_keep=None, save_relative_paths=True)

    @abc.abstractmethod
    def configureAtomicContributions(self):
        pass
