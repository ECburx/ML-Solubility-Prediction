from data.dataset import Dataset
from model.abstractmodel import AbstractModel


class GCN(AbstractModel):
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
            task_type: str,
            n_tasks: int,
            n_classes: int = None,
            graph_conv_layers=None,
            residual=True,
            batchnorm=False,
            dropout=0,
            self_loop=True,
            device=None,
            **kwargs
    ):
        """
        :param task_type: Regression or Classification
        :param n_tasks: Number of tasks.
        :param n_classes: Number of classes.
        :param graph_conv_layers: Number of convolution layers.
        :param residual: Whether to use residual connection in GCN layers.
        :param batchnorm: Whether to apply batch normalization is on the output of the GCN layer
        :param dropout: dropout probability on the output of GCN
        :param self_loop: If graphs contain self-loop.
        :param device: torch.device
        :param kwargs: Other parameters
        """
        super().__init__(task_type, description_info="DeepChem_MolGraphConvFeaturizer")

        if graph_conv_layers is None:
            graph_conv_layers = [64, 64]

        from deepchem.models import GCNModel
        self.model = GCNModel(
            n_tasks=n_tasks,
            graph_conv_layers=graph_conv_layers,
            residual=residual,
            batchnorm=batchnorm,
            dropout=dropout,
            mode=task_type,
            n_classes=n_classes,
            self_loop=self_loop,
            device=device,
            **kwargs
        )

    def fit(
            self,
            dataset: Dataset,
            *args,
            epochs: int = 30,
            X_name: str = "graph",
            y_name: str = "sol_category",
            **kwargs):
        import deepchem as dc
        deepchem_ds = dc.data.NumpyDataset(
            X=dataset.X[X_name].tolist(),
            y=dataset.y[y_name].tolist()
        )
        self.model.fit(
            deepchem_ds,
            nb_epoch=epochs,
            **kwargs
        )

    def save(self, path):
        pass

    def load(self, path):
        pass

    def predict(self,
                dataset: Dataset,
                X_name: str = "graph"):
        import deepchem as dc
        deepchem_ds = dc.data.NumpyDataset(
            X=dataset.X[X_name].tolist(),
        )
        return self.model.predict(deepchem_ds)
