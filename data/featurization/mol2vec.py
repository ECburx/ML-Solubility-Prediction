from data.featurization.base import BaseRepresentation
import pandas as pd


class Mol2Vec(BaseRepresentation):
    """
    Mol2Vec provides fingerprints for ML approaches to learn vector representations of molecular substructures,
    inspired by natural language processing techniques.

    Jaeger, S., Fulle, S. and Turk, S. (2018). Mol2vec: Unsupervised Machine Learning Approach with Chemical Intuition.
    Journal of Chemical Information and Modeling, 58(1), pp.27–35.
    doi:https://doi.org/10.1021/acs.jcim.7b00616.
    """

    def convert(self, Xs, ys=None):
        from deepchem.feat import Mol2VecFingerprint
        return pd.DataFrame(Mol2VecFingerprint().featurize(Xs.to_list()))
