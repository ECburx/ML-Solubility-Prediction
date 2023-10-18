from data.featurization.base import BaseRepresentation
import pandas as pd


class DeepChem_MolGraphConvFeaturizer(BaseRepresentation):
    """
    DeepChem Graph Representations.
    For Graph convolutions.
    """

    def convert(self, Xs, ys=None):
        from deepchem.feat import MolGraphConvFeaturizer
        return pd.DataFrame({
            "graph": MolGraphConvFeaturizer(
                use_edges=True
            ).featurize(Xs.iloc[:, -1].to_list())
        })
