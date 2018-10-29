from core import AtomicEnergyPotential, nn_layer, precision
from utils import calculate_bp_maps
import tensorflow as _tf

class BPAtomicNN():
    def __init__(self, input_dim, layers = [20], offset = 0,
        act_funs = [_tf.nn.tanh]):
        self.input = _tf.placeholder(shape = (None, input_dim),
            dtype = precision, name = "ANN_input")
        self.input = _tf.placeholder(shape = (None, input_dim),
            dtype = precision, name = "ANN_input")

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
        for (t, dims, lays, offs, acts) in zip(self.atom_types, input_dims,
            layers, offsets, act_funs):
            with _tf.variable_scope("{}_ANN".format(t), reuse = _tf.AUTO_REUSE):
                self.atomic_contributions[t] = BPAtomicNN(dims, lays, offs, acts)
