from ase.build import molecule
from xtb.ase.calculator import XTB
from ase.optimize import LBFGS
from ase.neb import NEB
from ase import io

initial = molecule('C2H6')
initial.set_calculator(XTB(method="GFN2-xTB"))

# Optimize initial geometry
opt = LBFGS(initial)
opt.run()

# Generate final position by permuting hydrogen positions
final = initial.copy()
final.positions[2:5] = initial.positions[[3, 4, 2]]

# Build list from copies of the initial geometry
images = [initial]
images += [initial.copy() for i in range(5)]
images += [final]

neb = NEB(images)
# Generate sensible inital path using the IDPP method
neb.interpolate(method='idpp')

# Save the results for later examples
io.write('ethane.traj', images)
# Write the gif for the documentation
io.write('ethane.gif', images, rotation='20x,20y')
