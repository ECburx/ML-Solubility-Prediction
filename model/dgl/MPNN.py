from model.dgl.DGLModel import DGLModel


class MPNN(DGLModel):
    def __init__(
            self,
            n_tasks: int,
            node_in_feats: int,
            edge_in_feats: int,
            node_out_feats: int = 64,
            edge_hidden_feats: int = 128,
            num_step_message_passing: int = 6,
            num_step_set2set: int = 6,
            num_layer_set2set: int = 3,
            **kwargs
    ):
        from dgllife.model import MPNNPredictor
        super().__init__(
            model=MPNNPredictor(
                node_in_feats=node_in_feats,
                edge_in_feats=edge_in_feats,
                node_out_feats=node_out_feats,
                edge_hidden_feats=edge_hidden_feats,
                n_tasks=n_tasks,
                num_step_message_passing=num_step_message_passing,
                num_step_set2set=num_step_set2set,
                num_layer_set2set=num_layer_set2set
            ),
            **kwargs
        )

    def _predict(self, graphs):
        node_feats = graphs.ndata.pop('h').to(self.device)
        edge_feats = graphs.edata.pop('e').to(self.device)
        return self.model(graphs, node_feats, edge_feats)

    def reset_parameters(self):
        self.model.gnn.reset_parameters()
