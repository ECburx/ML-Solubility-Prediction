from model.dgl.DGLModel import DGLModel
import torch


class PAGTN(DGLModel):
    def __init__(
            self,
            n_tasks: int,
            node_in_feats: int,
            node_out_feats: int,
            node_hid_feats: int,
            edge_feats: int,
            depth: int = 5,
            nheads: int = 1,
            dropout: float = 0.1,
            activation: torch.nn.functional = torch.nn.functional.leaky_relu,
            mode: str = "mean",
            **kwargs
    ):
        assert mode in ["mean", "max", "sum"]
        from dgllife.model import PAGTNPredictor
        super().__init__(
            model=PAGTNPredictor(
                node_in_feats=node_in_feats,
                node_out_feats=node_out_feats,
                node_hid_feats=node_hid_feats,
                edge_feats=edge_feats,
                depth=depth,
                nheads=nheads,
                dropout=dropout,
                activation=activation,
                n_tasks=n_tasks,
                mode=mode
            ),
            **kwargs
        )

    def _predict(self, graphs):
        node_feats = graphs.ndata.pop('h').to(self.device)
        edge_feats = graphs.edata.pop('e').to(self.device)
        return self.model(graphs, node_feats, edge_feats)
