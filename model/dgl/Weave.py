import torch.nn

from model.dgl.DGLModel import DGLModel


class Weave(DGLModel):
    def __init__(
            self,
            task_type: str,
            n_tasks: int,
            node_in_feats: int,
            edge_in_feats: int,
            num_gnn_layers: int,
            gnn_hidden_feats: int = 50,
            gnn_activation: torch.nn.functional = torch.nn.functional.relu,
            graph_feats: int = 50,
            gaussian_expand: bool = True,
            gaussian_memberships: list[tuple[float]] = None,
            readout_activation: torch.nn.modules.activation = torch.nn.Tanh(),
            **kwargs
    ):
        if gaussian_memberships is None:
            gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134),
                                    (-0.468, 0.118), (-0.228, 0.114), (0., 0.114),
                                    (0.228, 0.114), (0.468, 0.118), (0.739, 0.134), (1.080, 0.170), (1.645, 0.283)]
        from dgllife.model.model_zoo.weave_predictor import WeavePredictor
        super().__init__(
            model=WeavePredictor(
                node_in_feats=node_in_feats,
                edge_in_feats=edge_in_feats,
                num_gnn_layers=num_gnn_layers,
                gnn_hidden_feats=gnn_hidden_feats,
                gnn_activation=gnn_activation,
                graph_feats=graph_feats,
                gaussian_expand=gaussian_expand,
                gaussian_memberships=gaussian_memberships,
                readout_activation=readout_activation,
                n_tasks=n_tasks
            ),
            **kwargs
        )

    def _predict(self, graphs):
        node_feats = graphs.ndata.pop('h').to(self.device)
        edge_feats = graphs.edata.pop('e').to(self.device)
        return self.model(graphs, node_feats, edge_feats)
