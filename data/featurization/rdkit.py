from data.featurization.base import BaseRepresentation
import pandas as pd


class Rdkit2D(BaseRepresentation):
    """
    A collection of chem-informatics and ML software written in C++ and Python, which also provides a series of two-
    and three-dimensional descriptors/fingerprints.

    Landrum G. RDKit: open-source cheminformatics. http://www.rdkit.org
    """

    def convert(self, Xs, ys=None):
        from olorenchemengine.representations import DescriptastorusDescriptor
        df = pd.DataFrame(DescriptastorusDescriptor(name="rdkit2d").convert(Xs))
        df.columns = [c[0] for c in DescriptastorusDescriptor(name="rdkit2d").generator.columns]
        return df


class Rdkit2D_Normalized(BaseRepresentation):
    """
    A collection of chem-informatics and ML software written in C++ and Python, which also provides a series of two-
    and three-dimensional descriptors/fingerprints.
    Descriptors are normalized.

    Landrum G. RDKit: open-source cheminformatics. http://www.rdkit.org
    """

    def convert(self, Xs, ys=None):
        from olorenchemengine.representations import DescriptastorusDescriptor
        df = pd.DataFrame(DescriptastorusDescriptor(name="rdkit2dnormalized").convert(Xs))
        df.columns = [c[0] for c in DescriptastorusDescriptor(name="rdkit2dnormalized").generator.columns]
        return df


class Rdkit_Mol(BaseRepresentation):
    def convert(self, Xs, ys=None):
        from rdkit.Chem import MolFromSmiles
        return pd.DataFrame({"mol": [MolFromSmiles(X) for X in Xs.iloc[:, 0].tolist()]})
