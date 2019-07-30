from ase.build import molecule
from ase.neb import NEB
from ase.calculators.emt import EMT
from ase.optimize.fire import FIRE
from mlpot.mlneb import run_mla_neb
from mlpot.calculators.gprcalculator import GPRCalculator
from mlpot.kernels import RBFKernel

#Optimise molecule
initial = molecule('C2H6')
initial.set_calculator(EMT())
relax = FIRE(initial)
relax.run(fmax=0.05)

#Create final state
final = initial.copy()
final.positions[2:5] = initial.positions[[3, 4, 2]]

#Generate blank images
images = [initial]

for i in range(9):
    images.append(initial.copy())

images.append(final)

for image in images:
    image.set_calculator(EMT())

#Run IDPP interpolation
neb = NEB(images)
neb.interpolate('idpp')

kernel = RBFKernel(constant=100.0, length_scale=1.0)
ml_calc = GPRCalculator(kernel=kernel, C1=1E8, C2=1E8, opt_restarts=1)

run_mla_neb(neb, ml_calc)
