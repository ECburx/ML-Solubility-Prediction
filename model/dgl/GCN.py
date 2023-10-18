from model.dgl.DGLModel import DGLModel
import torch


class GCN(DGLModel):
    """
    Graph Convolutional Networks

    The Graphical Convolutional Network (GCN) is a scalable approach presented by Kipf and Welling (2016) for
    semi-supervised learning on graph-structured data. It is based on a variant of convolutional neural networks that
    operate directly on graphs.

    Similar to the convolution operation, convolution in GCNs involves the model learning features by examining
    neighbouring nodes. If we consider that CNNs are designed to operate on Euclidean structured data, GNNs can be
    considered the generalized version of CNNs, as they can handle varying numbers of node connections and unordered
    nodes on irregular or non-Euclidean structured data.

    Given an undirected graph $\mathcal{G}$, Kipf and Welling (2016) proposed a multi-layer GCN that employs the
    following layer-wise propagation rule

    $$
    \textbf{H}^{(l)} = \sigma \left( \tilde{\textbf{D}}^{-\frac{1}{2}} \tilde{\textbf{A}} \tilde{\textbf{D}}^{-\frac{1}{2}} \textbf{H}^{(l-1)} \textbf{W}^{(l-1)} \right)
    $$

    and the definition of the spectral convolution of a signal $x$ with a filter $g_{\theta'}$ at each iteration
     $k \in K$

    $$
    g_{\theta'} \star x \approx \sum^K_{k=0} \theta'_k T_k (\tilde{\textbf{L}})x
    $$

    where

    - $\textbf{H}^{(l)}$ is the matrix of node representations at layer $l$
    - $\sigma$ is the activation function
    - $\textbf{W}$ is the trainable parameter matrix
    - the Chebyshev polynomials are recursively defined as $T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$
    - $\theta' \in \mathbb{R}^K$ is a vector of Chebyshev coefficients
    - given the identity matrix $\textbf{I}$ and the normalized graph Laplacian $\textbf{L}$
        - $\tilde{\textbf{A}} = \textbf{A} + \textbf{I}$ is the graph adjacency matrix with added self-connections
        - $\tilde{\textbf{D}}_{ii} = \sum_j \tilde{\textbf{A}} _{ij}$
        - $\tilde{\textbf{L}} = \frac{2}{\lambda_{\text{max}}}\textbf{L} - \textbf{I}$ (here $\lambda_{\text{max}}$
          denotes the largest eigenvalue of $\textbf{L}$)
    """

    def __init__(
            self,
            n_tasks: int,
            in_feats: int,
            hidden_feats: list[int],
            gnn_norm: str,  # 'right', 'both', 'none'
            activation: torch.nn.functional = torch.nn.functional.relu,
            residual: bool = True,
            batchnorm: bool = False,
            dropout: float = 0,
            classifier_hidden_feats: int = 128,
            classifier_dropout: float = 0,
            predictor_hidden_feats: int = 128,
            predictor_dropout: int = 0,
            **kwargs
    ):
        assert gnn_norm in ['right', 'both', 'none']
        layer_num = len(hidden_feats)

        from dgllife.model import GCNPredictor
        super().__init__(
            model=GCNPredictor(
                in_feats=in_feats,
                hidden_feats=hidden_feats,
                gnn_norm=[gnn_norm] * layer_num,
                n_tasks=n_tasks,
                activation=[activation] * layer_num,
                residual=[residual] * layer_num,
                batchnorm=[batchnorm] * layer_num,
                dropout=[dropout] * layer_num,
                classifier_hidden_feats=classifier_hidden_feats,
                classifier_dropout=classifier_dropout,
                predictor_hidden_feats=predictor_hidden_feats,
                predictor_dropout=predictor_dropout
            ),
            **kwargs
        )

    def reset_parameters(self):
        self.model.gnn.reset_parameters()
