from NeuralNetworks.SymmetryFunctionSet import SymmetryFunctionSet_py as SymFunSet_py, SymmetryFunctionSet as SymFunSet_cpp
from NeuralNetworks.SymmetryFunctionSetC import SymmetryFunctionSet as SymFunSet_c
import numpy as np
import matplotlib.pyplot as plt

plot_derivatives = False

sfs_cpp = SymFunSet_cpp(["Ni", "Au"], cutoff = 100000.)
sfs_py = SymFunSet_py(["Ni", "Au"], cutoff = 100000.)
sfs_c = SymFunSet_c(["Ni", "Au"], cutoff = 100000.)

pos = np.array([[0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0,-1.0, 0.0]])
types = ["Ni", "Ni", "Au", "Ni", "Au"]

rss = [0.0]
etas = [np.log(2.0)]

sfs_cpp.add_radial_functions(rss, etas)
sfs_cpp.add_angular_functions([1.0], [1.0], etas)
sfs_py.add_radial_functions(rss, etas)
sfs_py.add_angular_functions([1.0],[1.0], etas)
sfs_c.add_radial_functions(rss, etas)
sfs_c.add_angular_functions([1.0],[1.0], etas)


geo = [(t, p) for t,p in zip(types, pos)]
out_cpp = sfs_cpp.eval_geometry(geo)
out_py = sfs_py.eval_geometry(geo)

print("Evalutation difference smaller 1e-6: {}".format(
    all(abs(np.array(out_cpp) - out_py).flatten() < 1e-6)))

analytical_derivatives = sfs_cpp.eval_derivatives(types, pos)
## Calculate numerical derivatives
numerical_derivatives = np.zeros((len(out_cpp), out_cpp[0].size, pos.size))
dx = 0.00001
for i in xrange(pos.size):
    dpos = np.zeros(pos.shape)
    dpos[np.unravel_index(i,dpos.shape)] += dx
    numerical_derivatives[:,:,i] = (np.array(sfs_cpp.eval(types, pos+dpos))
        - np.array(sfs_cpp.eval(types, pos-dpos)))/(2*dx)

print("Derivatives difference smaller 1e-6: {}".format(all(
    abs(numerical_derivatives.flatten() -
    np.array(analytical_derivatives).flatten()) < 1e-6)))

if plot_derivatives:
    fig, ax = plt.subplots(ncols = 3, figsize = (12,4));
    p1 = ax[0].pcolormesh(numerical_derivatives)
    plt.colorbar(p1, ax = ax[0])
    p2 = ax[1].pcolormesh(analytical_derivatives)
    plt.colorbar(p2, ax = ax[1])
    p3 = ax[2].pcolormesh(numerical_derivatives - analytical_derivatives)
    plt.colorbar(p3, ax = ax[2])
    plt.show()
