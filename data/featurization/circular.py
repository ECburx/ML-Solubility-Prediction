from data.featurization.base import BaseRepresentation
import pandas as pd


class Circular(BaseRepresentation):
    """
    Extended-connectivity fingerprints (ECFPs) are a novel class of topological fingerprints for molecular
    characterization.

    Rogers, D. and Hahn, M. (2010). Extended-Connectivity Fingerprints.
    Journal of Chemical Information and Modeling, 50(5), pp.742–754.
    doi:https://doi.org/10.1021/ci100050t.
    """

    def convert(self, Xs, ys=None):
        from deepchem.feat import CircularFingerprint
        return pd.DataFrame(CircularFingerprint().featurize(Xs.to_list()))
