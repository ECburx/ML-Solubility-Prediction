from data.featurization.base import BaseRepresentation
import pandas as pd


class SPGNN_Graph(BaseRepresentation):
    def convert(self, Xs, ys=None):
        from olorenchemengine.external.SPGNN.main import SPGNN_PYG
        return pd.DataFrame({
                "graph": SPGNN_PYG().convert(
                        Xs=Xs,
                        ys=ys.iloc[:, -1] if ys is not None else None)
        })
