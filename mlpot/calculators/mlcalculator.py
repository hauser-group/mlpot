from ase.calculators.calculator import (Calculator, all_changes)


class MLCalculator(Calculator):
    """Base class for all machine learning calculators"""

    implemented_properties = ['energy', 'forces']

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, C1=1.0, C2=1.0, **kwargs):
        Calculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms, **kwargs)

        self.C1 = C1
        self.C2 = C2
        self.atoms_train = []

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results['energy'], self.results['forces'] = self.predict(atoms)

    def add_data(self, atoms):
        self.atoms_train.append(atoms)

    def fit(self):
        pass

    def predict(self, atoms):
        pass

    def get_params(self):
        pass

    def set_params(self, **params):
        pass
