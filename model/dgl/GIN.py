from model.dgl.DGLModel import DGLModel


class GIN(DGLModel):
    """
    Graph Isomorphism and Jumping Knowledge

    Xu et al. (2018) posit that a GNN with maximal discriminative power could distinguish between distinct graph
    structures by mapping them onto unique representations in the embedding space. To achieve this, they proposed the
    Graph Isomorphism Network (GIN), a neural network architecture that extends the Weisfeiler-Lehman test. By doing so,
    they could ensure that isomorphic graphs were mapped onto identical representations while non-isomorphic graphs were
    mapped onto distinct ones. Furthermore, they have demonstrated that the GIN achieves the greatest discriminative
    power among all GNNs. Subsequently, Hu et al. (2019) augmented the GIN model with a pre-training strategy to
    mitigate negative transfer effects in downstream tasks, leading to promising results on benchmark datasets.

    Given a pair of graphs $\mathcal{G}_1$ and $\mathcal{G}_2$, Hamilton (2023) summerized the goal of graph isomorphism
    testing is to declare whether or not these two graphs are isomorphic, formally, if or not the following conditions
    hold

    $$
    \textbf{P}\textbf{A}_1\textbf{P}^{\top} = \textbf{A}_2 \text{ and } \textbf{P}\textbf{X}_1 = \textbf{X}_2
    $$

    where $\textbf{P}$ is a permutation matrix, $\textbf{A}_1$ and $\textbf{A}_2$ are adjacency matrices of
    $\mathcal{G}_1$ and $\mathcal{G}_2$, as well as node features $\textbf{X}_1$ and $\textbf{X}_2$.

    Jumping Knowledge (JK) (Xu et al., 2018) is a network architecture that can be combined with GIN, providing an
    adaptive means to adjust the influence radii of each node and task through the selective combination of various
    aggregations at the final layer. In other words, it breaks the assumption of utilizing the output of the final layer
    of the GNN. Thus, it could facilitate improved structure-aware representations. In this approch, Hamilton (2023)
    defined the final node representations $\textbf{z}_u$ as

    $$
    \textbf{z}_u = f_{\text{JK}} \left( \textbf{h}_u^{(0)} \oplus \textbf{h}_u^{(1)} \oplus \cdots \oplus \textbf{h}_u^{(K)} \right)
    $$

    where $f_{\text{JK}}$ is an arbitrary differentiable function.

    In our study, we employed DGL-LifeSci and RdKit to integrate the pre-training strategies of GIN and JK. To perform
    featurization of chemical compounds, we one-hot encoded atoms and their corresponding chiralities into graphs, which
    also accounted for four types of chemical bonds (single, double, triple, and aromatic) and three directions of bonds
    (end upright, end downright, and none).
    """

    def __init__(
            self,
            n_tasks: int,
            pretrained: str,
            num_node_emb_list: list[int],
            num_edge_emb_list: list[int],
            num_layers: int = 5,
            emb_dim: int = 300,
            JK: str = "last",
            dropout: float = 0.5,
            readout: str = "mean",
            **kwargs
    ):
        """
        :param task_type: Regression or Classification
        :param n_tasks: Number of tasks.
        :param pretrained: Whether to use pretrained parameters
        :param num_node_emb_list: the number of items to embed for the categorical node feature variables
        :param num_edge_emb_list: the number of items to embed for the categorical edge feature variables
        :param num_layers: Number of GIN layers to use
        :param emb_dim: The size of each embedding vector
        :param JK: jumping knowledge
        :param dropout:
        :param readout:
        :param kwargs:
        """
        assert pretrained in ["gin_supervised_contextpred", "gin_supervised_infomax",
                              "gin_supervised_edgepred", "gin_supervised_masking"]
        assert JK in ["concat", "last", "max", "sum"]
        assert readout in ["sum", "mean", "max", "attention", "set2set"]

        from dgllife.model import GINPredictor
        super().__init__(
            model=GINPredictor(
                num_node_emb_list=num_node_emb_list,
                num_edge_emb_list=num_edge_emb_list,
                num_layers=num_layers,
                emb_dim=emb_dim,
                JK=JK,
                dropout=dropout,
                readout=readout,
                n_tasks=n_tasks
            ),
            **kwargs
        )
        self.model.gnn.JK = JK

    def _predict(self, graphs):
        categorical_node_feats = [
            graphs.ndata.pop('atomic_number').to(self.device),
            graphs.ndata.pop('chirality_type').to(self.device)
        ]
        categorical_edge_feats = [
            graphs.edata.pop('bond_type').to(self.device),
            graphs.edata.pop('bond_direction_type').to(self.device)
        ]
        return self.model(graphs, categorical_node_feats, categorical_edge_feats)

    def reset_parameters(self):
        self.model.gnn.reset_parameters()
