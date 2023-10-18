from abc import ABC, abstractmethod


class BaseRepresentation(ABC):
    """
    Abstract Representations of SMIELS.
    """

    @abstractmethod
    def convert(self, Xs, ys=None):
        """
        Convert SMILES in Xs to their corresponding descriptors.
        :param Xs: SMIELS
        :param ys: Labels. (Optional)
        :return: Featurized SMILES.
        """
        pass
