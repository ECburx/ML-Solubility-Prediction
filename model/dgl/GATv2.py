from model.dgl.DGLModel import DGLModel
import torch


class GATv2(DGLModel):
    """
    GATv2

    After GAT, Brody et al. (2021) illustrate a confined form of attention that is computed by the graph attention
    network (GAT) - the ranking of the attention scores is unconditioned on the query node. This constrained form of
    attention is formally defined as static attention and is distinguished from a more flexible form of attention called
    dynamic attention. The static attention mechanism impairs the ability of GAT to accurately model the training data.
    To overcome this limitation, the authors propose GATv2 in 2022, a variant of dynamic graph attention that offers
    enhanced expressive power compared to GAT. The fixed version of the graph attention mechanism becomes

    $$
    \alpha_{un} = \frac{
    \text{exp} \left( \vec{\textbf{a}}^\textbf{T} \text{LeakyReLu} \left(  \textbf{W} \left[ u || n \right] \right) \right)
    }{
    \sum_{v\in \mathcal{N}(u)} \text{exp} \left( \vec{\textbf{a}}^\textbf{T} \text{LeakyReLu} \left( \textbf{W} \left[ u || v \right] \right) \right) }
    $$

    which makes a significant difference in the expressiveness of the attention function - *A GATv2 layer computes
    dynamic attention for any set of node representations $\mathbb{K} = \mathbb{Q} = \mathcal{V}$* Brody et al. (2021)
    where $\mathbb{K}$ is a set of key vectors and $\mathbb{Q}$ is a set of query vectors.
    """
    def __init__(
            self,
            n_tasks: int,
            in_feats: int,
            hidden_feats: list[int],
            num_heads: int = 4,
            feat_drops: float = 0,
            attn_drops: float = 0,
            alphas: float = 0.2,
            residuals: bool = False,
            activations: torch.nn.functional = torch.nn.functional.relu,
            allow_zero_in_degree: bool = False,
            biases: bool = True,
            share_weights: bool = False,
            agg_modes: str = "mean",  # "flatten", "mean"
            predictor_out_feats: int = 128,
            predictor_dropout: float = 0,
            get_attention: bool = False,
            **kwargs
    ):
        """
        :param task_type: Regression or Classification
        :param n_tasks: Number of tasks.
        :param in_feats: Number of input features.
        :param hidden_feats: Number of hidden features.
        :param num_heads: Number of attention heads.
        :param feat_drops: Dropout applied to the input features
        :param attn_drops: Dropout applied to attention values of edges
        :param alphas: Hyperparameter in LeakyReLU, which is the slope for negative values.
        :param residuals: Whether to perform skip connection in GAT layer, default to True.
        :param activations: Activation function.
        :param allow_zero_in_degree: Allow zero in degree?
        :param biases: Whether to use bias in the GAT layer.
        :param share_weights: if the learnable weight matrix for source and destination nodes is the same in the layer.
        :param agg_modes: aggregate multi-head attention results in GAT layers.
        """
        layer_num = len(hidden_feats)
        self.get_attention = get_attention

        from dgllife.model.model_zoo.gatv2_predictor import GATv2Predictor
        super().__init__(
            model=GATv2Predictor(
                in_feats=in_feats,
                hidden_feats=hidden_feats,
                num_heads=[num_heads] * layer_num,
                feat_drops=[feat_drops] * layer_num,
                attn_drops=[attn_drops] * layer_num,
                alphas=[alphas] * layer_num,
                residuals=[residuals] * layer_num,
                activations=[activations] * layer_num,
                allow_zero_in_degree=allow_zero_in_degree,
                biases=[biases] * layer_num,
                share_weights=[share_weights] * layer_num,
                agg_modes=[agg_modes] * layer_num,
                n_tasks=n_tasks,
                predictor_dropout=predictor_dropout,
                predictor_out_feats=predictor_out_feats
            ),
            **kwargs
        )

    def _predict(self, graphs):
        node_feats = graphs.ndata.pop('h').to(self.device)
        return self.model(graphs, node_feats, self.get_attention)

    def reset_parameters(self):
        self.model.gnn.reset_parameters()
