import numpy as np

class DTree:

    def __init__(self):
        pass

    def convert(self, db):
        pass

    def counter(self, data: list):
        """
        Counts elements in the list and returns a dictionary
        with the elements as keys the amount as values
        """
        dct = dict()

        for element in data:
            if element in dct:
                dct[element] += 1
            else:
                dct[element] = 1

        return dct

    def entropy(self, numbers: list):
        total = sum(numbers)
        entro = 0
        for n in numbers:
            posibility = n/total
            entro -= posibility * np.log2(posibility)

        return entro


    # Pass by category to define the children
    def infogain(self, parent, child1, child2):
        return self.entropy(parent) - self.entropy((child1 + child2)/2)