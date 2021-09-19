from ase.io import read
from ase.calculators.emt import EMT
from ase.neb import NEB
from ase.optimize import FIRE

images = read('diffusion.traj', index=':')

for image in images:
    image.set_calculator(EMT())

neb = NEB(images)

opt = FIRE(neb)
opt.run()
