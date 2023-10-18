from data.featurization.base import BaseRepresentation
import pandas as pd


class PubChem(BaseRepresentation):
    """
    PubChem fingerprints is a 881 bit structural key, which is used by PubChem for similarity searching.

    PubChem Substructure Fingerprint. (n.d.). Available at:
    https://web.cse.ohio-state.edu/~zhang.10631/bak/drugreposition/list\_fingerprints.pdf [Accessed 12 Feb. 2023].
    """

    def convert(self, Xs, ys=None):
        from deepchem.feat import PubChemFingerprint
        return pd.DataFrame(PubChemFingerprint().featurize(Xs.iloc[:, -1].to_list()))
