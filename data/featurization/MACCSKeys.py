from data.featurization.base import BaseRepresentation
import pandas as pd


class MACCSKeys(BaseRepresentation):
    """
    Optimized MDL Keys provide optimized MDL keysets to improve the performance for use in molecular similarity.

    Durant, J.L., Leland, B.A., Henry, D.R. and Nourse, J.G. (2002). Reoptimization of MDL Keys for Use in Drug
    Discovery. Journal of Chemical Information and Computer Sciences, 42(6), pp.1273–1280.
    doi:https://doi.org/10.1021/ci010132r.
    """

    def convert(self, Xs, ys=None):
        from olorenchemengine.representations import MACCSKeys
        return pd.DataFrame(MACCSKeys().convert(Xs))
