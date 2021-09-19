from ase import io
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.neb import NEB

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
initial = fcc100('Al', size=(3, 2, 3))
# Replace two Al atoms by Pt:
initial[12].symbol = 'Pt'
initial[15].symbol = 'Pt'
add_adsorbate(initial, 'Au', 1.7, 'hollow')
initial.center(axis=2, vacuum=4.0)

# Fix all substrate atoms.
mask = [atom.tag > 0 for atom in initial]
constraint = FixAtoms(mask=[atom.tag > 0 for atom in initial])
initial.set_constraint(constraint)

# Use EMT potential:
initial.set_calculator(EMT())

# Initial state:
qn = QuasiNewton(initial)
qn.run(fmax=0.05)

# Final state:
final = initial.copy()
final.set_calculator(EMT())
final[-1].x += final.get_cell()[0, 0] / 3
qn = QuasiNewton(final)
qn.run(fmax=0.05)

images = [initial]
for i in range(5):
    image = initial.copy()
    image.set_calculator(EMT())
    image.set_constraint(constraint)
    images.append(image)
images.append(final)

neb = NEB(images)
neb.interpolate()

io.write('diffusion.traj', images)
io.write('diffusion.gif', images, rotation='-80x,-30y,-5z')
