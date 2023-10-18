from data.featurization.MACCSKeys import MACCSKeys
from data.featurization.circular import Circular
from data.featurization.mol2vec import Mol2Vec
from data.featurization.mordred import Mordred
from data.featurization.pubchem import PubChem
from data.featurization.rdkit import Rdkit2D_Normalized
from data.featurization.base import BaseRepresentation
import pandas as pd


class ADMET(BaseRepresentation):
    """
    The absorption, distribution, metabolism, excretion, and toxicity (ADMET) properties are important in drug discovery
    as they define efficacy and safety.

    The paper used six featurizers from DeepChem to compute fingerprints and descriptors:
    -   MACCS fingerprints are common structural keys that compute a binary string based on a molecule’s structural features.
    -   Extended-connectivity circular fingerprints compute a bit vector by breaking up a molecule into circular neighborhoods. They are widely used for structure-activity modeling.
    -   Mol2Vec fingerprints create vector representations of molecules based on an unsupervised machine learning approach.
    -   PubChem fingerprints consist of 881 structural keys that cover a wide range of substructures and features. They are used by PubChem for similarity searching.
    -   Mordred descriptors calculate a set of chemical descriptors such as the count of aromatic atoms or the count of all halogen atoms.
    -   RDKit descriptors calculate a set of chemical descriptors such as molecular weight and the number of radical electrons.

    Reference:
        Tian, H., Ketkar, R. & Tao, P. ADMETboost: a web server for accurate ADMET prediction. J Mol Model 28,
        408 (2022).
        https://doi.org/10.1007/s00894-022-05373-8
    """
    descriptors = [
            MACCSKeys(),
            Circular(),
            Mol2Vec(),
            PubChem(),
            Mordred(),
            Rdkit2D_Normalized()
    ]

    def convert(self, Xs, ys=None):
        results = [d.convert(Xs) for d in ADMET.descriptors]
        return pd.concat(results, axis=1, join="inner")
