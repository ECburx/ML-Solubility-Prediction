from model.dgl.DGLModel import DGLModel
import torch


class GAT(DGLModel):
    """
    # GAT

    Velickovic et al. (2017) introduced a novel neural network architecture, known as the graph attention network (GAT),
    for processing graph-structured data. GATs overcome the limitations of previous graph convolution-based methods by
    utilizing masked self-attentional layers. Through layer stacking, nodes are empowered to attend to the features of
    their neighborhoods, allowing for the implicit assignment of varying weights to different nodes without the need for
    expensive matrix operations or prior knowledge of the graph structure. Given an input graph
    $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ and a set of node features
    $\textbf{X} \in \mathbb{R}^{d \times |\mathcal{V}|}$,
    Velickovic et al. (2017) describe their graph attention mechanism as follows.

    $$
    \alpha_{un} = \frac{
    \text{exp} \left( \text{LeakyReLu} \left( \vec{\textbf{a}}^\textbf{T} \left[ \textbf{W}_u || \textbf{W}_n \right] \right) \right)
    }{
    \sum_{v\in \mathcal{N}(u)} \text{exp} \left( \text{LeakyReLu} \left( \vec{\textbf{a}}^\textbf{T} \left[ \textbf{W}_u || \textbf{W}_v \right] \right) \right) }
    $$

    where

    - for each node $u \in \mathcal{V}$, $v \in \mathcal{N}(u)$ and $\mathcal{N}(u)$ is the set of neighbourhood nodes
      of $u$
    - $\alpha_{un}$ is the attention coefficient that indicates the importance of node $n$'s features to node $u$
    - LeakyReLu is the activation function
    - $.^T$ represents transposition
    - $||$ is the concatenation operator
    - $\vec{\textbf{a}}$ is a weight vector that parameterise the attention mechanism $a$, where $a$ can be a neural
      network

    The graph structure is injected into the mechanism by performing *masked attention*, which computes
    $a ( \textbf{W}_u, \textbf{W}_v )$ for nodes $v \in \mathcal{N}(u) $.
    The normalized attention coefficients are employed to compute a linear combination of the corresponding features,
    thereby producing the final output features for each node. To enhance the learning process of self-attention,
    a multi-head attention mechanism is utilized.

    $$
    u' = \text{\LARGE $||$}^M_{m=1} \sigma \left( \sum_{v \in \mathcal{N}(u)} \alpha_{uv}^{(m)} \textbf{W}^{(m)}_v \right)
    $$

    where

    - $\sigma$ is the potential nonlinearity
    - $M$ is the number of independent attention mechanisms to execute
    - $\alpha_{uv}^{(m)}$ are normalized attention coefficients computed by the $m^{\text{th}}$ attention mechanism
      $a^{(m)}$ and $W^{(m)}$ is the corresponding input linear transformation's weight matrix.
    """

    def __init__(
            self,
            n_tasks: int,
            in_feats: int,
            hidden_feats: list[int],
            num_heads: int,
            feat_drops: float = 0,
            attn_drops: float = 0,
            alphas: float = 0.2,
            residuals: bool = True,
            agg_modes: str = "mean",  # "mean", "flatten"
            activation: torch.nn.functional = torch.nn.functional.relu,
            biases: bool = True,
            classifier_hidden_feats: int = 128,
            classifier_dropout: float = 0,
            predictor_hidden_feats: int = 128,
            predictor_dropout: int = 0,
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
        :param agg_modes: The way to aggregate multi-head attention results in a GAT, can be either 'flatten' for
        concatenating all-head results or 'mean' for averaging all head results.
        :param activation: Activation function.
        :param biases: Whether to use bias in the GAT layer.
        :param kwargs: Other parameters
        """
        from dgllife.model import GATPredictor
        layer_num = len(hidden_feats)

        super().__init__(
            model=GATPredictor(
                in_feats=in_feats,
                hidden_feats=hidden_feats,
                num_heads=[num_heads] * layer_num,
                feat_drops=[feat_drops] * layer_num,
                attn_drops=[attn_drops] * layer_num,
                alphas=[alphas] * layer_num,
                residuals=[residuals] * layer_num,
                agg_modes=[agg_modes] * layer_num,
                activations=[activation] * layer_num,
                biases=[biases] * layer_num,
                classifier_hidden_feats=classifier_hidden_feats,
                classifier_dropout=classifier_dropout,
                n_tasks=n_tasks,
                predictor_hidden_feats=predictor_hidden_feats,
                predictor_dropout=predictor_dropout
            ),
            **kwargs
        )

    def reset_parameters(self):
        self.model.gnn.reset_parameters()
