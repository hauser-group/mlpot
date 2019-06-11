from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(3, 2, 3))
# Replace two Al atoms by Pt:
slab[12].symbol = 'Pt'
slab[15].symbol = 'Pt'
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)

# Fix all substrate atoms. 
mask = [atom.tag > 0 for atom in slab]
#print(mask)
slab.set_constraint(FixAtoms(mask=mask))

# Use EMT potential:
slab.set_calculator(EMT())

# Initial state:
qn = QuasiNewton(slab, trajectory='initial.traj')
qn.run(fmax=0.05)

# Final state:
slab[-1].x += slab.get_cell()[0, 0] / 3
qn = QuasiNewton(slab, trajectory='final.traj')
qn.run(fmax=0.05)
