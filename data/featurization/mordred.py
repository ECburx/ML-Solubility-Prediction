from data.featurization.base import BaseRepresentation
import pandas as pd


class Mordred(BaseRepresentation):
    """
    Mordred: a molecular descriptor calculator which can calculate more than 1800 two- and three-dimensional
    descriptors, even for large molecules, which cannot be accomplished by other software.

    Moriwaki, H., Tian, YS., Kawashita, N. et al. Mordred: a molecular descriptor calculator. J Cheminform 10, 4 (2018).
    https://doi.org/10.1186/s13321-018-0258-y
    """

    def convert(self, Xs, ys=None, ignore_3D: bool = False):
        from deepchem.feat import MordredDescriptors
        from mordred import Calculator, descriptors
        df = pd.DataFrame(MordredDescriptors(ignore_3D=ignore_3D).featurize(Xs.to_list()))
        df.columns = Calculator(descriptors, ignore_3D=ignore_3D).descriptors
        return df
