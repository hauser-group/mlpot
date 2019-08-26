from ase.build import molecule
from ase.neb import NEB
from ase.calculators.emt import EMT
from ase.optimize.fire import FIRE

# Optimise molecule
initial = molecule('C2H6')
initial.set_calculator(EMT())
relax = FIRE(initial)
relax.run(fmax=0.05)

# Create final state
final = initial.copy()
final.positions[2:5] = initial.positions[[3, 4, 2]]

# Generate blank images
images = [initial]

for i in range(9):
    images.append(initial.copy())

images.append(final)

for image in images:
    image.set_calculator(EMT())

# Run IDPP interpolation
neb = NEB(images)
neb.interpolate('idpp')

# Run NEB calculation
opt = FIRE(neb)
opt.run(fmax=0.05)
