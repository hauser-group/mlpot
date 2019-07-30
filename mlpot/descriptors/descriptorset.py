import cDescriptorSet


class DescriptorSet:

    def __init__(self, atomtypes, cutoff=7.0):
        self.atomtypes = atomtypes
        self.cutoff = cutoff
        self.type_dict = {}
        self.num_Gs = [0]*len(atomtypes)
        for i, t in enumerate(atomtypes):
            self.type_dict[t] = i
            self.type_dict[i] = i
        self.descriptorsetCapsule = cDescriptorSet.construct(len(atomtypes))

    def __delete__(self):
        cDescriptorSet.delete_object(self.descriptorsetCapsule)
