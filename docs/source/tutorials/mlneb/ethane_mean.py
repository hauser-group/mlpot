from ase import io
from xtb.ase.calculator import XTB
from ase.neb import NEB
from ase.optimize import FIRE
from mlpot.mlneb import run_mla_neb
from mlpot.calculators.gprcalculator import GPRCalculator
from mlpot.kernels import RBFKernel
from ase.calculators.emt import EMT

images = io.read('images.traj', index=':')

for image in images:
    image.set_calculator(XTB(method="GFN2-xTB"))

neb = NEB(images, remove_rotation_and_translation=True)

kernel = RBFKernel(constant=100.0, length_scale=1.0)
ml_calc = GPRCalculator(kernel=kernel, C1=1e8, C2=1e8, opt_restarts=1,
                        mean_model=EMT())

# Pass custom optimizer to mute optimization
opt = lambda atoms: FIRE(atoms, logfile=None)
run_mla_neb(neb, ml_calc, optimizer=opt)

