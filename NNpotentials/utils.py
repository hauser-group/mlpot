import tensorflow as _tf
import numpy as _np

def morse(x, lamb = 0.23314621043800202, alpha = 0.6797779934458726):
    return lamb*(_tf.exp(-2.0*alpha*(x))-2.0*_tf.exp(-alpha*(x)))

def stash(x, lamb = 1.1613855392326946, alpha = 0.6520042387583171):
    return _tf.where(x <= 0.0, 2.0*_tf.tanh(alpha*x),
	   lamb*_tf.asinh(2.0*alpha*x/lamb))

def stash_old(x, lamb = 1.1613326990732873, alpha = 0.6521334159737763):
    return _tf.where(x <= 0.0, 2.0*_tf.tanh(alpha*x),
	   lamb*_tf.asinh(2.0*alpha*x/lamb))

def calculate_eam_maps(num_atom_types, _Gs, _types):
    batchsize = len(_Gs)
    r = [[[] for _ in range(num_atom_types)] for _ in range(num_atom_types)]
    b_indices = [[[] for _ in range(num_atom_types)] for _ in range(num_atom_types)]
    Ns = [0]*num_atom_types
    indices = [[] for _ in range(num_atom_types)]

    for i, (G_vec, t_vec) in enumerate(zip(_Gs, _types)):
        for Gi, ti in zip(G_vec, t_vec):
            indices[ti].append([i, Ns[ti]])
            for tj in range(num_atom_types):
                for j in range(len(Gi[tj])):
                    b_indices[ti][tj].append([Ns[ti], len(r[ti][tj])+j])
                r[ti][tj].extend(Gi[tj])
            Ns[ti] += 1

    # Cast into numpy arrays, also takes care of wrong dimensionality of empty
    # lists
    maps = []
    b_maps = [[[] for _ in range(num_atom_types)] for _ in range(num_atom_types)]
    for i in range(num_atom_types):
        indices[i] = _np.array(indices[i], dtype = _np.int64).reshape((-1,2))
        maps.append(_tf.SparseTensorValue(indices[i], [1.0]*Ns[i], [batchsize, Ns[i]]))
        for j in range(num_atom_types):
             b_indices[i][j] = _np.array(b_indices[i][j], dtype = _np.int64).reshape((-1,2))
             b_maps[i][j] = _tf.SparseTensorValue(b_indices[i][j], [1.0]*len(r[i][j]), [Ns[i], len(r[i][j])])
             r[i][j] = _np.array(r[i][j]).reshape((-1,1))
    return r, b_maps, maps

def calculate_bp_maps(num_atom_types, _Gs, _types):
    batchsize = len(_Gs)
    Ns = [0]*num_atom_types
    indices = [[] for _ in range(num_atom_types)]
    atoms = [[] for _ in range(num_atom_types)]

    for i, (G_vec, t_vec) in enumerate(zip(_Gs, _types)):
        for Gi, ti in zip(G_vec, t_vec):
            indices[ti].append([i, Ns[ti]])
            atoms[ti].append(Gi)
            Ns[ti] += 1

    # Cast into numpy arrays, also takes care of wrong dimensionality of empty
    # lists
    maps = []
    for a in range(num_atom_types):
         indices[a] = _np.array(indices[a], dtype = _np.int64).reshape((-1,2))
         maps.append(_tf.SparseTensorValue(indices[a], [1.0]*Ns[a], [batchsize, Ns[a]]))
         atoms[a] = _np.array(atoms[a])
    return atoms, maps

def calculate_bp_indices(num_atom_types, Gs, types, dGs = None):
    indices = [[] for _ in range(num_atom_types)]
    atoms = [[] for _ in range(num_atom_types)]

    if dGs is None:
        for i, (G_vec, t_vec) in enumerate(zip(Gs, types)):
            for Gi, ti in zip(G_vec, t_vec):
                atoms[ti].append(Gi)
                indices[ti].append(i)

        # Cast into numpy arrays, also takes care of wrong dimensionality of
        # empty lists
        for a in range(num_atom_types):
            indices[a] = _np.array(indices[a], dtype = _np.int32).reshape((-1,1))
            atoms[a] = _np.array(atoms[a])
        return atoms, indices
    else:
        atom_derivs = [[] for _ in range(num_atom_types)]
        for i, (G_vec, t_vec, dG_vec) in enumerate(zip(Gs, types, dGs)):
            for Gi, ti, dGi in zip(G_vec, t_vec, dG_vec):
                atoms[ti].append(Gi)
                atom_derivs[ti].append(dGi)
                indices[ti].append(i)

        # Cast into numpy arrays, also takes care of wrong dimensionality of
        # empty lists
        for a in range(num_atom_types):
            indices[a] = _np.array(indices[a], dtype = _np.int32).reshape((-1,1))
            atoms[a] = _np.array(atoms[a])
            atom_derivs[a] = _np.array(atom_derivs[a])
        return atoms, indices, atom_derivs
