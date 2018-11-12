from .core import AtomicEnergyPotential, nn_layer, precision
from .utils import calculate_bp_maps
import tensorflow as _tf

class BPAtomicNN():
    def __init__(self, input_tensor, layers = [20], offset = 0,
        act_funs = [_tf.nn.tanh]):
        self.input = input_tensor

        # Start of with input layer as previous layer
        previous = self.input
        self.layers = []
        for i, (n, act) in enumerate(zip(layers, act_funs)):
            previous, _, _ = nn_layer(previous, previous.shape[-1].value,
                n, name = "hiddenLayer_%d"%(i+1), act = act)
            self.layers.append(previous)
        self.output, _, _ = nn_layer(previous, previous.shape[-1].value,
            1, act = None, initial_bias = [offset], name = "outputLayer")
        self.layers.append(self.output)


class BPpotential(AtomicEnergyPotential):
    def __init__(self, atom_types, input_dims, layers = None, offsets = None,
        act_funs = None):
        with _tf.variable_scope("BPpot"):
            if layers == None:
                layers = [[20]]*len(atom_types)
            if offsets == None:
                offsets = [0.0]*len(atom_types)
            if act_funs == None:
                act_funs = []
                for lays in layers:
                    act_funs.append([_tf.nn.tanh]*len(lays))
            AtomicEnergyPotential.__init__(self, atom_types,
                input_dims = input_dims, layers = layers, offsets = offsets,
                act_funs = act_funs)

    def configureAtomicContributions(self, **kwargs):
        input_dims = kwargs.get('input_dims')
        layers = kwargs.get('layers')
        offsets = kwargs.get('offsets')
        act_funs = kwargs.get('act_funs')
        for (t, in_dim, lays, offs, acts) in zip(self.atom_types, input_dims,
            layers, offsets, act_funs):
            with _tf.variable_scope("{}_ANN".format(t), reuse = _tf.AUTO_REUSE):
                input_tensor = _tf.placeholder(shape = (None, in_dim),
                    dtype = precision, name = "ANN_input")
                self.atomic_contributions[t] = BPAtomicNN(
                    input_tensor, lays, offs, acts)

def build_BPestimator(atom_types, input_dims, layers = None, offsets = None,
    act_funs = None):
    if layers == None:
        layers = [[20]]*len(atom_types)
    if offsets == None:
        offsets = [0.0]*len(atom_types)
    if act_funs == None:
        act_funs = []
        for lays in layers:
            act_funs.append([_tf.nn.tanh]*len(lays))

    feature_columns = []
    for i, a in enumerate(atom_types):
        feature_columns.append(
            _tf.feature_column.numeric_column(key='%s_input'%a, shape=(input_dims[i])))
        feature_columns.append(
            _tf.feature_column.numeric_column(key='%s_indices'%a, dtype=_tf.int32))
    def model_fun(features, labels, mode, params):
        atomic_contributions = {}
        atom_types = params['atom_types']
        for (t, lays, offs, acts) in zip(atom_types,
            params['layers'], params['offsets'], params['act_funs']):
            with _tf.variable_scope("{}_ANN".format(t), reuse = _tf.AUTO_REUSE):
                input_tensor = features['%s_input'%t]
                atomic_contributions[t] = BPAtomicNN(
                    input_tensor, lays, offs, acts)

        predicted_energies = _tf.scatter_nd(
            _tf.concat([features['%s_indices'%t] for t in atom_types], 0),
            _tf.concat([_tf.reshape(atomic_contributions[t].output, [-1])
            for t in atom_types], 0), _tf.shape(labels),
            name = "E_prediction")

        if mode == _tf.estimator.ModeKeys.PREDICT:
            predictions = {'energies': predicted_energies}
            return _tf.estimator.EstimatorSpec(mode, predictions=predictions)

        num_atoms = _tf.reduce_sum([_tf.bincount(features['%s_indices'%t])
            for t in atom_types], axis = 0, name = "NumberOfAtoms")
        # Compute loss.
        loss = _tf.losses.mean_squared_error(
            labels=labels, predictions=predicted_energies)

        rmse = _tf.metrics.root_mean_squared_error(labels, predicted_energies)
        metrics = {'rmse': rmse}
        _tf.summary.scalar('rmse', rmse[1])

        if mode == _tf.estimator.ModeKeys.EVAL:
            return _tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        assert mode == _tf.estimator.ModeKeys.TRAIN
        optimizer = _tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=_tf.train.get_global_step())
        return _tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    estimator = _tf.estimator.Estimator(model_fn = model_fun,
        params={'feature_columns':feature_columns,
                'atom_types':atom_types,
                'layers':layers,
                'offsets':offsets,
                'act_funs':act_funs})

    return estimator
