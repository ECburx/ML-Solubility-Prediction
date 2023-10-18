from model.dgl.DGLModel import DGLModel
import torch


class AlphaGNN(DGLModel):
    """
    We studied the success of AlphaFold 2 and attempted to incorporate its architectural design into our previously
    described model ingeniously. AlphaFold 2 incorporates neural network architectures and training procedures that are
    guided by the evolutionary, physical, and geometric constraints of protein structures.

    We chose to integrate GAT and GCN layers as the attention-based and non-attention-based components, respectively,
    in our subnetwork called AlphaNN. The aim is to take advantage of the respective strengths of both models.
    Specifically, GAT layers are proficient in modeling the node-to-node relationships in the graph,
    while GCN layers are well-suited for capturing the global graph structure.

    Another notable technique utilized in their model involves reinforcing the concept of iterative refinement, termed
    recycling, which could be integrated into our solubility prediction model, such as AlphaNN and 1D-CNN.
    """

    def __init__(
            self,
            task_type: str,
            n_tasks: int,
            in_feats: int,
            recycle: int,
            allow_zero_in_degree: bool = False,
            activation=torch.nn.functional.relu,
            gat_num_heads: int = 2,
            gat_feat_drop: float = 0.,
            gat_attn_drop: float = 0.,
            gat_alpha: float = 0.2,
            gat_residual: bool = False,
            gat_agg_mode: str = "flatten",
            gat_bias: bool = False,
            gcn_norm: str = "both",
            gcn_residual: bool = True,
            gcn_batchnorm: bool = True,
            gcn_dropout: float = 0.,
            recycle_alpha: float = 0.5,
            predictor_hidden_feats: int = 128,
            predictor_dropout: float = 0.,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            **kwargs
    ):
        """
        :param task_type: Regression or Classification
        :param n_tasks: Number of tasks.
        :param in_feats: Number of input features.
        :param recycle: Number of recycling.
        :param allow_zero_in_degree: If allow zero in degree.
        :param activation: Activation function.
        :param gat_num_heads: Number of Heads in a GAT layer.
        :param gat_feat_drop: Dropout applied to the input features
        :param gat_attn_drop: Dropout applied to attention values of edges
        :param gat_alpha: Hyperparameter in LeakyReLU, which is the slope for negative values.
        :param gat_residual: Whether to perform skip connection in GAT layer, default to True.
        :param gat_agg_mode: The way to aggregate multi-head attention results in a GAT, can be either 'flatten' for
        concatenating all-head results or 'mean' for averaging all head results.
        :param gat_bias: Whether to use bias in the GAT layer.
        :param gcn_norm: Whether to use batch normalization
        :param gcn_residual: Whether to use residual connection in GCN layers.
        :param gcn_batchnorm: Whether to apply batch normalization is on the output of the GCN layer
        :param gcn_dropout: dropout probability on the output of GCN
        :param recycle_alpha: Alpha value of recycling
        :param predictor_hidden_feats: Hidden feats in the predictor.
        :param predictor_dropout: Dropout of predictor
        :param device: torch.device
        :param kwargs: Other parameters
        """
        super().__init__(
            model=AlphaGNNModel(
                n_tasks=n_tasks,
                in_feats=in_feats,
                recycle=recycle,
                allow_zero_in_degree=allow_zero_in_degree,
                activation=activation,
                gat_num_heads=gat_num_heads,
                gat_feat_drop=gat_feat_drop,
                gat_attn_drop=gat_attn_drop,
                gat_alpha=gat_alpha,
                gat_residual=gat_residual,
                gat_agg_mode=gat_agg_mode,
                gat_bias=gat_bias,
                gcn_norm=gcn_norm,
                gcn_residual=gcn_residual,
                gcn_batchnorm=gcn_batchnorm,
                gcn_dropout=gcn_dropout,
                recycle_alpha=recycle_alpha,
                predictor_hidden_feats=predictor_hidden_feats,
                predictor_dropout=predictor_dropout,
                device=device
            ),
            task_type=task_type,
            description_info="DGL_Graph",
            device=device,
            **kwargs
        )

    def reset_parameters(self):
        self.model.reset_parameters()


class AlphaGNNModel(torch.nn.Module):
    def __init__(
            self,
            n_tasks: int,
            in_feats: int,
            recycle: int,
            device,
            allow_zero_in_degree: bool = False,
            activation=torch.nn.functional.relu,
            gat_num_heads: int = 2,
            gat_feat_drop: float = 0.,
            gat_attn_drop: float = 0.,
            gat_alpha: float = 0.2,
            gat_residual: bool = False,
            gat_agg_mode: str = "flatten",
            gat_bias: bool = False,
            gcn_norm: str = "both",
            gcn_residual: bool = True,
            gcn_batchnorm: bool = True,
            gcn_dropout: float = 0.,
            recycle_alpha: float = 0.5,
            predictor_hidden_feats: int = 128,
            predictor_dropout: float = 0.
    ):
        super().__init__()

        layers = []
        for i in range(recycle):
            layer = AlphaGNNLayer(
                in_feats=in_feats,
                allow_zero_in_degree=allow_zero_in_degree,
                activation=activation,
                gat_num_heads=gat_num_heads,
                gat_feat_drop=gat_feat_drop,
                gat_attn_drop=gat_attn_drop,
                gat_alpha=gat_alpha,
                gat_residual=gat_residual,
                gat_agg_mode=gat_agg_mode,
                gat_bias=gat_bias,
                gcn_norm=gcn_norm,
                gcn_residual=gcn_residual,
                gcn_batchnorm=gcn_batchnorm,
                gcn_dropout=gcn_dropout,
                device=device
            )
            layers.append(layer)

        self.gnn = AlphaGNNRecycle(
            layers=layers,
            alpha=recycle_alpha
        )

        from dgllife.model.readout import WeightedSumAndMax
        from dgllife.model.model_zoo.mlp_predictor import MLPPredictor
        self.readout = WeightedSumAndMax(in_feats=in_feats)
        self.predict = MLPPredictor(
            in_feats=2 * in_feats,
            hidden_feats=predictor_hidden_feats,
            n_tasks=n_tasks,
            dropout=predictor_dropout
        )

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, bg, feats):
        feats = self.gnn(bg, feats)
        feats = self.readout(bg, feats)
        return self.predict(feats)


class AlphaGNNLayer(torch.nn.Module):
    def __init__(
            self,
            in_feats: int,
            device,
            allow_zero_in_degree: bool = False,
            activation=torch.nn.functional.relu,
            gat_num_heads: int = 2,
            gat_feat_drop: float = 0.,
            gat_attn_drop: float = 0.,
            gat_alpha: float = 0.2,
            gat_residual: bool = False,
            gat_agg_mode: str = "flatten",
            gat_bias: bool = False,
            gcn_norm: str = "both",
            gcn_residual: bool = True,
            gcn_batchnorm: bool = True,
            gcn_dropout: float = 0.
    ):
        super().__init__()
        assert gat_agg_mode in ["flatten", "mean"]
        assert gcn_norm in ['right', 'both', 'none']

        from dgllife.model.gnn.gat import GATLayer
        from dgllife.model.gnn.gcn import GCNLayer

        self.gat = GATLayer(
            in_feats=in_feats,
            out_feats=in_feats,
            num_heads=gat_num_heads,
            feat_drop=gat_feat_drop,
            attn_drop=gat_attn_drop,
            alpha=gat_alpha,
            residual=gat_residual,
            agg_mode=gat_agg_mode,
            activation=activation,
            bias=gat_bias,
            allow_zero_in_degree=allow_zero_in_degree
        ).to(device)

        if gat_agg_mode == "flatten":
            hidden_feats = in_feats * gat_num_heads
        else:
            hidden_feats = in_feats

        self.gcn = GCNLayer(
            in_feats=hidden_feats,
            out_feats=in_feats,
            gnn_norm=gcn_norm,
            activation=activation,
            residual=gcn_residual,
            batchnorm=gcn_batchnorm,
            dropout=gcn_dropout,
            allow_zero_in_degree=allow_zero_in_degree
        ).to(device)

    def reset_parameters(self):
        self.gat.reset_parameters()
        self.gcn.reset_parameters()

    def forward(self, g, feats):
        feats = self.gat(g, feats)
        feats = self.gcn(g, feats)
        return feats


class AlphaGNNRecycle(torch.nn.Module):
    def __init__(
            self,
            layers: list[AlphaGNNLayer],
            alpha: float = 0.5
    ):
        super().__init__()
        self.layers = layers
        self.alpha = alpha

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, feats):
        for layer in self.layers:
            # feats = layer(g, feats) + feats
            feats = self.alpha * layer(g, feats) + (1 - self.alpha) * feats
            # feats = layer(g, feats) * feats
        return feats
