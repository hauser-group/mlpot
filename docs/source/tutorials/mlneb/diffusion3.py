from ase.io import read, write
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.neb import NEB
from ase.optimize import BFGS
from mlpot.mlneb import run_mla_neb
from mlpot.calculators.gprcalculator import GPRCalculator
from mlpot.kernels import RBFKernel

initial = read('initial.traj')
final = read('final.traj')

constraint = FixAtoms(mask=[atom.tag > 0 for atom in initial])

images = [initial]
for i in range(5):
    image = initial.copy()
    image.set_calculator(EMT())
    image.set_constraint(constraint)
    images.append(image)

images.append(final)

neb = NEB(images)
neb.interpolate()

kernel = RBFKernel(constant=100.0, length_scale=1.0)
ml_calc = GPRCalculator(kernel=kernel, C1=1E8, C2=1E8, opt_restarts=1)

run_mla_neb(neb, ml_calc)
write('minimum_energy_path.traj', images)
