import numpy as np

from data.featurization.base import BaseRepresentation
import pandas as pd


class DGL_Graph(BaseRepresentation):
    """
    DGL-LifeSci provides built-in support for constructing three kinds of graphs for molecules – molecular graphs,
    distance-based graphs, and complete graphs.

    Edges in a molecular graph correspond to chemical bounds, while every pair of atoms is connected in a complete
    graph. As for a distance-based graph, an edge between a pair of atoms is constructed if the distance between them
    is within a cutoff distance.

    Generated graphs must be fed into DGL-constructed GNNs for further operation.

    Fey, M. and Lenssen, J.E. (2019). Fast Graph Representation Learning with PyTorch Geometric.
    arXiv:1903.02428 [cs, stat].
    doi: https://doi.org/10.48550/arXiv.1903.02428.
    """

    def __init__(
            self,
            graph_type: str,
            featurize_type: str = None,
            self_loop: bool = True,
            node_featurizer=None,
            edge_featurizer=None
    ):
        """
        :param graph_type: GraphType. DEFAULT: bi-directed DGLGraph
        :param self_loop: DEFAULT: FALSE
        :param node_featurizer: {CanonicalAtomFeaturizer (DEFAULT), AttentiveFPAtomFeaturizer}
        :param edge_featurizer: {CanonicalBondFeaturizer (DEFAULT), AttentiveFPBondFeaturizer}
        :param featurize_type:
        """
        assert graph_type in ["COMPLETE", "BI_GRAPH"]
        if featurize_type is not None:
            assert featurize_type in ["Canonical", "AttentiveFP", "Pretrain"]
        else:
            assert node_featurizer is not None and edge_featurizer is not None

        self.graph_type = graph_type
        self.self_loop = self_loop

        if featurize_type == "Canonical":
            from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
            self.node_featurizer = CanonicalAtomFeaturizer()
            self.edge_featurizer = CanonicalBondFeaturizer(self_loop=self_loop)
        elif featurize_type == "AttentiveFP":
            from dgllife.utils import AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
            self.node_featurizer = AttentiveFPAtomFeaturizer()
            self.edge_featurizer = AttentiveFPBondFeaturizer(self_loop=self_loop)
        elif featurize_type == "Pretrain":
            from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
            self.node_featurizer = PretrainAtomFeaturizer()
            self.edge_featurizer = PretrainBondFeaturizer(self_loop=self_loop)
        else:
            self.node_featurizer = node_featurizer
            self.edge_featurizer = edge_featurizer

    def convert(self, Xs, ys=None):
        return np.array([self.smiles_to_graph(smiles) for smiles in Xs.to_list()])

    def get_node_feat_size(self):
        return self.node_featurizer.feat_size()

    def get_edge_feat_size(self):
        return self.edge_featurizer.feat_size()

    def smiles_to_graph(self, smiles: str):
        """
        Convert a SMILES into a DGLGraph and featurize for it.
        :param smiles: SMILES
        :return: DGLGraph representing specified SMILES
        """

        graph = None

        if self.graph_type == "COMPLETE":
            from dgllife.utils import smiles_to_complete_graph
            graph = smiles_to_complete_graph(
                smiles,
                add_self_loop=self.self_loop,
                node_featurizer=self.node_featurizer,
                edge_featurizer=self.edge_featurizer
            )
        elif self.graph_type == "BI_GRAPH":
            from dgllife.utils import smiles_to_bigraph
            graph = smiles_to_bigraph(
                smiles,
                add_self_loop=self.self_loop,
                node_featurizer=self.node_featurizer,
                edge_featurizer=self.edge_featurizer
            )

        return graph
