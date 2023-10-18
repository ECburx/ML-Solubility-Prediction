import pandas as pd

from data.featurization.base import BaseRepresentation


class OGB_Geometric_Graph(BaseRepresentation):
    """
    OGB Geometric Graph Representation.
    """

    def convert(self, Xs, ys=None):
        from olorenchemengine import TorchGeometricGraph, OGBAtomFeaturizer, OGBBondFeaturizer

        return pd.DataFrame({
            "graph": TorchGeometricGraph(
                atom_featurizer=OGBAtomFeaturizer(),
                bond_featurizer=OGBBondFeaturizer()
            ).convert(
                Xs=Xs,
                ys=ys.iloc[:, -1] if ys is not None else None
            )
        })
