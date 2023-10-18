from model.dgl.DGLModel import DGLModel


class AttentiveFP(DGLModel):
    """
    AttentiveFP

    The AttentiveFP is introduced in 2020, a graph-based neural network framework that captures the atomic local
    environment by propagating node information from neighbouring nodes to distant ones but also enables nonlocal
    effects at the intramolecular level by employing a graph attention mechanism. Their findings demonstrate that the
    AttentiveFP can effectively extract nonlocal intramolecular interactions that are often challenging to model using
    traditional graph-based representations. The notion of incorporating an attention mechanism into graph-based models
    aims to derive a context vector for the target node by emphasizing its neighboring nodes and local environment.
    Specifically, the Attentive FP molecular representation approach employs a dual-stack mechanism of attentive layers
    to extract information from the molecular graph. The first stack is responsible for atom embedding, while the second
    is for full-molecule embedding.

    The attention mechanism is incorporated into the individual-atom and full-molecule embedding processes. To create
    the molecule embedding, all atom embeddings are aggregated through a super virtual node, interconnecting all atoms
    within the molecule. In terms of atom embedding, a graph attention mechanism is introduced at each layer to
    assimilate information from the surrounding neighbourhoods.

    Based on the multi-head attention mechanism from GAT, the attention context $n'$ is combined with the current state
    vector of the target atom and fed into a recurrent gated unit (GRU). The GRU recurrent network unit facilitates
    efficient transmission of information to the surrounding nodes during the update iterations.
    """

    def __init__(
            self,
            n_tasks: int,
            node_feat_size: int,
            edge_feat_size: int,
            graph_feat_size: int,
            num_layers: int = 2,
            num_timesteps: int = 2,
            dropout: float = 0,
            **kwargs
    ):
        """
        :param task_type: Regression or Classification
        :param n_tasks: Number of tasks.
        :param node_feat_size: Number of input node features.
        :param edge_feat_size: Number of input edge features.
        :param graph_feat_size: Number of input features in a graph.
        :param num_layers: Number of layers.
        :param num_timesteps: Number of time steps.
        :param dropout: Dropout.
        :param kwargs: Other parameters
        """
        from dgllife.model import AttentiveFPPredictor
        super().__init__(
            model=AttentiveFPPredictor(
                node_feat_size=node_feat_size,
                edge_feat_size=edge_feat_size,
                num_layers=num_layers,
                num_timesteps=num_timesteps,
                graph_feat_size=graph_feat_size,
                n_tasks=n_tasks,
                dropout=dropout
            ),
            **kwargs
        )

    def _predict(self, graphs):
        node_feats = graphs.ndata.pop('h').to(self.device)
        edge_feats = graphs.edata.pop('e').to(self.device)
        return self.model(graphs, node_feats, edge_feats)

    def reset_parameters(self):
        self.model.gnn.reset_parameters()
