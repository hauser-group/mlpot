from ase.io.xyz import simple_write_xyz
import os
import numpy as np
from ase.calculators.calculator import FileIOCalculator
import ase.units


class XTB_custom(FileIOCalculator):
    implemented_properties = ['energy', 'forces']
    command = 'xtb PREFIX.xyz -grad > PREFIX.xtbo'

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='xtb', atoms=None, **kwargs):

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        if os.path.isfile("charges"):
            os.remove("charges")
        if os.path.isfile("gradient"):
            os.remove("gradient")
        if os.path.isfile("energy"):
            os.remove("energy")
        if os.path.isfile("xtbrestart"):
            os.remove("xtbrestart")
        with open(self.label + '.xyz', 'w') as fout:
            simple_write_xyz(fout, [atoms])

    def read_results(self):
        energy = 0.0
        gradient = []
        with open('gradient') as fin:
            for line in fin:
                sp = line.split()
                if sp[0] == "cycle":
                    energy = float(sp[6]) * ase.units.Hartree
                if len(sp) == 3:
                    gradient.append([float(sp[0].replace('D', 'E')),
                                     float(sp[1].replace('D', 'E')),
                                     float(sp[2].replace('D', 'E'))])
        self.results = {'energy': energy,
                        'forces': -np.array(gradient) * ase.units.Hartree}
