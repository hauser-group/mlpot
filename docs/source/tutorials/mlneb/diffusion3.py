from ase.io import read
from ase.calculators.emt import EMT
from ase.neb import NEB
from mlpot.mlneb import run_mla_neb
from mlpot.calculators.gprcalculator import GPRCalculator
from mlpot.kernels import RBFKernel

images = read('diffusion.traj', index=':')

for image in images:
    image.set_calculator(EMT())

neb = NEB(images)

kernel = RBFKernel(constant=100.0, length_scale=1.0)
ml_calc = GPRCalculator(kernel=kernel, C1=1e8, C2=1e8, opt_restarts=1)

run_mla_neb(neb, ml_calc)
