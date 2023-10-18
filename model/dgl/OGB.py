from model.dgl.DGLModel import DGLModel
import torch


class OGB(DGLModel):
    def __init__(
            self,
            n_tasks: int,
            in_edge_feats: int,
            num_node_types: int = 1,
            hidden_feats: int = 300,
            n_layers: int = 5,
            batchnorm: bool = True,
            activation: torch.nn.functional = torch.nn.functional.relu,
            dropout: float = 0,
            gnn_type: str = "gcn",
            virtual_node: bool = True,
            residual: bool = True,
            jk: bool = False,
            readout: str = "mean",
            **kwargs
    ):
        assert gnn_type in ["gcn", "gin"]
        assert readout in ["mean", "sum", "max"]
        from dgllife.model import GNNOGBPredictor
        super().__init__(
            model=GNNOGBPredictor(
                in_edge_feats=in_edge_feats,
                num_node_types=num_node_types,
                hidden_feats=hidden_feats,
                n_layers=n_layers,
                n_tasks=n_tasks,
                batchnorm=batchnorm,
                activation=activation,
                dropout=dropout,
                gnn_type=gnn_type,
                virtual_node=virtual_node,
                residual=residual,
                jk=jk,
                readout=readout
            ),
            **kwargs
        )

    def _predict(self, graphs):
        node_feats = graphs.ndata.pop('h').long().to(self.device)
        edge_feats = graphs.edata.pop('e').to(self.device)
        return self.model(graphs, node_feats, edge_feats)
