from ase import io
from xtb.ase.calculator import XTB
from ase.neb import NEB
from ase.optimize import FIRE

images = io.read('images.traj', index=':')

for image in images:
    image.set_calculator(XTB(method="GFN2-xTB"))

neb = NEB(images, remove_rotation_and_translation=True)

optimizer = FIRE(neb)
optimizer.run()
