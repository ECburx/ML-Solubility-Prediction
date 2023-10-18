import copy

import pandas as pd

from data.dataset import Dataset
from model.abstractmodel import AbstractModel
import torch.cuda
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error


class CNN1D(AbstractModel):
    """
    Inspired by the second place solution of the Mechanisms of Action (MoA) Prediction Challange
    https://www.kaggle.com/c/lish-moa/discussion/202256,
    we develop a model that based on 1D convolutional layers. The dimensions of the input features are initially
    expanded to $2^{12}$ using a fully connected layer, followed by reshaping into segmented sequences. This procedure
    enables the model to have sufficient capacity to execute subsequential operations and learn patterns. Subsequently,
    four convolutional components are concatenated. A pooling operation is performed between the first two convolutional
    layers to decrease the dimensions of the hidden layer by consolidating the outputs of neuron clusters in the
    previous layer into a single neuron in the subsequent layer. The output of the second convolutional component is
    connected to the last convolutional layer through production as a shortcut mechanism.

    In each convolutional component, the signal sequences are first batch normalized to enhance the training process's
    efficiency and stability by re-centring and re-scaling. The subsequential dropout layer helps avoid overfitting by
    scaling inputs not set to 0 up by $1/(1 - \text{rate})$ such that the sum over all inputs remains constant. A
    Rectified Linear Unit (ReLU) layer is chosen as the activation function. Similar to the 2D convolution, the key
    operation of the component, the computation of a neuron in a 1D convolution operation, can also be described as

    $$
    \hat{y}_{i,k} = B_k + \sum_{u=0}^{f_d-1} { \sum_{k'=0}^{f_{n'}-1} { x_{i',k'} \cdot w_{u,k',k} } } \ \text{with} \ i' = i \times s_d + u
    $$

    where

    - $\hat{y}_{i,k}$ is the output of the neuron in position $i$ and feature map $k$ in the convolutional layer $l$
    - $x_{i',k'}$ is the output of the neuron in position $i'$ and the convolutional layer $l-1$
    - $B_k$ is the bias term for feature map $k$ in the convolutional layer $l$
    - $s_d$ is the strides
    - $f_d$ is the length of the strides
    - $f_{n'}$ is the number of feature maps in the convolutional layer $l-1$
    - $\cdot w_{u,k',k}$ is the connection weight between any neuron in feature map $k$ of the convolutional layer $l$
      and its input located at position $u$ and feature map $k'$
    """

    def __init__(
            self,
            task_type: str,
            n_tasks: int,
            in_feats: int,
            lr: float = 0.001,
            weight_decay=0,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            **kwargs
    ):
        """
        :param task_type: Regression or Classification
        :param n_tasks: Number of tasks.
        :param in_feats: Number of input features.
        :param lr: Learning rate.
        :param weight_decay: Weight Decay
        :param device: torch.device
        :param kwargs: Other parameters
        """
        super().__init__(task_type=task_type, description_info="1D Tabular Data")
        self.best_state_dict = None
        self.model_config = kwargs
        self.in_feats = in_feats
        self.out_feats = n_tasks
        self.device = device
        self.model = _TabCnn1d(
            in_feats=self.in_feats,
            out_feats=self.out_feats,
            **kwargs
        ).to(self.device)
        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_f = torch.nn.SmoothL1Loss()

    def fit(
            self,
            trn: Dataset,
            max_epochs: int,
            *args,
            min_epochs: int = 0,
            early_stop: int = 0,
            val: Dataset = None,
            batch_size: int = 128,
            verbose: bool = True,
            **kwargs
    ):
        """
        :param trn: Training set.
        :param max_epochs: Maximum number of epochs.
        :param min_epochs: Minimum number of epochs.
        :param early_stop: Early stopping patience.
        :param val: Validation set.
        :param batch_size: Batch size.
        :param verbose: Verbose
        :param kwargs: other parameters.
        :return: Validation scores in each epoch.
        """
        scores = {"loss": [], "val_loss": [], "val_rmse": []}
        stop_counter = early_stop
        best_rmse = float("inf")

        trn_dl = DataLoader(
            dataset=_Dataset(trn),
            batch_size=batch_size
        )
        val_dl = None if val is None else DataLoader(
            dataset=_Dataset(val),
            batch_size=batch_size
        )

        bar = None
        for e in (bar := tqdm(range(max_epochs))) if verbose else range(max_epochs):
            loss = self._train_epoch(trn_dl)
            scores["loss"].append(loss)

            if val is None:
                if bar is not None:
                    bar.set_postfix_str(f"loss: {loss:.3f}")
            else:
                val_loss, pred = self._validate_epoch(val_dl)
                val_rmse = mean_squared_error(val.y, pd.DataFrame(pred).fillna(0), squared=False)
                scores["val_loss"].append(val_loss)
                scores["val_rmse"].append(val_rmse)
                if val_rmse <= best_rmse:
                    best_rmse = val_rmse
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())

                if bar is not None:
                    bar.set_postfix_str(f"loss: {loss:.3f} val_loss: {val_loss:.3f} val_rmse: {val_rmse:.3f}")

                if e > min_epochs:
                    if val_rmse <= best_rmse:
                        stop_counter = early_stop
                    else:
                        stop_counter -= 1

            if early_stop > 0 >= stop_counter:
                break

        return scores

    def _train_epoch(self, dataloader):
        self.model.train()
        loss = 0
        for data in dataloader:
            self.optim.zero_grad()
            X, y = data
            X = X.to(self.device)
            y = y.to(self.device)
            l = self.loss_f(self.model(X), y)
            l.backward()
            self.optim.step()
            loss += l.item()
        return loss

    def _validate_epoch(self, dataloader):
        self.model.eval()
        loss = 0
        pred = []

        for data in dataloader:
            X, y = data
            X = X.to(self.device)
            y = y.to(self.device)
            pred_y = self.model(X)
            l = self.loss_f(self.model(X), y)
            loss += l.item()
            pred.append(pred_y.detach().cpu().numpy())

        return loss, np.concatenate(pred)

    def _predict_epoch(self, dataloader, use_best_state):
        if use_best_state:
            model = _TabCnn1d(
                in_feats=self.in_feats,
                out_feats=self.out_feats,
                **self.model_config
            )
            model.load_state_dict(self.best_state_dict)
            model = model.to(self.device)
        else:
            model = self.model

        model.eval()
        pred = []

        for data in dataloader:
            X, _ = data
            X = X.to(self.device)
            with torch.no_grad():
                pred.append(model(X).detach().cpu().numpy())

        return np.concatenate(pred)

    def predict(self, dataset: Dataset, batch_size: int = 128, use_best_state: bool = False):
        dl = DataLoader(
            dataset=_Dataset(dataset),
            batch_size=batch_size
        )
        return self._predict_epoch(dl, use_best_state)

    def cross_validate(self, dataset: Dataset, epochs: int, *args, **kwargs):
        pass


class _Dataset(TorchDataset):
    def __init__(self, dataset: Dataset):
        self.X = torch.tensor(dataset.X.values, dtype=torch.float)
        self.y = None if dataset.y is None else torch.tensor(dataset.y.values, dtype=torch.float)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], None if self.y is None else self.y[index]


class _TabCnn1d(torch.nn.Module):
    def __init__(
            self,
            in_feats: int,
            out_feats: int,
            dense_feats: int = 4096,
            dropout: float = 0.1,
            celu_alpha: float = 0.06,
            # Convolutional Layer 1
            conv1_channels: int = 256,
            conv1_kernel_size: int = 5,
            conv1_stride: int = 1,
            conv1_padding: int = 1,
            conv1_bias: bool = True,
            # Convolutional Layer 2
            conv2_channels: int = 512,
            conv2_kernel_size: int = 3,
            conv2_stride: int = 1,
            conv2_padding: int = 1,
            conv2_bias: bool = True,
            # Convolutional Layer 2-1
            conv2_1_kernel_size: int = 3,
            conv2_1_stride: int = 1,
            conv2_1_padding: int = 1,
            conv2_1_bias: bool = True,
            conv2_1_dropout: float = 0.3,
            # Convolutional Layer 2-2
            conv2_2_kernel_size: int = 5,
            conv2_2_stride: int = 1,
            conv2_2_padding: int = 2,
            conv2_2_bias: bool = True,
            conv2_2_dropout: float = 0.2,
            # Decoder
            conv_out_channels: int = 512,
            decoder_kernel_size: int = 4,
            decoder_stride: int = 2,
            decoder_padding: int = 1,
            decoder_dropout: float = 0.2
    ):
        super().__init__()
        self.conv1_channels = conv1_channels
        self.dense_feats = dense_feats

        from torch.nn import (Sequential, BatchNorm1d, Dropout, Linear, CELU, Conv1d,
                              AdaptiveAvgPool1d, ReLU, MaxPool1d, Flatten)
        from torch.nn.utils import weight_norm

        self.dense = Sequential(
            BatchNorm1d(in_feats),
            Dropout(dropout),
            weight_norm(
                Linear(
                    in_features=in_feats,
                    out_features=dense_feats
                )
            ),
            CELU(celu_alpha)
        )
        self.conv1 = Sequential(
            BatchNorm1d(conv1_channels),
            Dropout(dropout),
            weight_norm(
                Conv1d(
                    in_channels=conv1_channels,
                    out_channels=conv2_channels,
                    kernel_size=conv1_kernel_size,
                    stride=conv1_stride,
                    padding=conv1_padding,
                    bias=conv1_bias
                )
            ),
            AdaptiveAvgPool1d(output_size=int(dense_feats / conv1_channels / 2))
        )
        self.conv2 = Sequential(
            BatchNorm1d(conv2_channels),
            Dropout(dropout),
            weight_norm(
                Conv1d(
                    in_channels=conv2_channels,
                    out_channels=conv2_channels,
                    kernel_size=conv2_kernel_size,
                    stride=conv2_stride,
                    padding=conv2_padding,
                    bias=conv2_bias
                )
            ),
            ReLU()
        )
        self.conv2_1 = Sequential(
            BatchNorm1d(conv2_channels),
            Dropout(conv2_1_dropout),
            weight_norm(
                Conv1d(
                    in_channels=conv2_channels,
                    out_channels=conv2_channels,
                    kernel_size=conv2_1_kernel_size,
                    stride=conv2_1_stride,
                    padding=conv2_1_padding,
                    bias=conv2_1_bias
                )
            ),
            ReLU()
        )
        self.conv2_2 = Sequential(
            BatchNorm1d(conv2_channels),
            Dropout(conv2_2_dropout),
            weight_norm(
                Conv1d(
                    in_channels=conv2_channels,
                    out_channels=conv_out_channels,
                    kernel_size=conv2_2_kernel_size,
                    stride=conv2_2_stride,
                    padding=conv2_2_padding,
                    bias=conv2_2_bias
                )
            ),
            ReLU()
        )
        decoder_in = int(dense_feats / conv1_channels / 2 / 2) * conv_out_channels
        self.decoder = Sequential(
            MaxPool1d(
                kernel_size=decoder_kernel_size,
                stride=decoder_stride,
                padding=decoder_padding
            ),
            Flatten(),
            BatchNorm1d(decoder_in),
            Dropout(decoder_dropout),
            weight_norm(
                Linear(
                    in_features=decoder_in,
                    out_features=out_feats
                )
            )
        )

    def forward(self, x):
        x = self.dense(x)
        x = x.reshape(x.size(0), self.conv1_channels, int(self.dense_feats / self.conv1_channels))
        x = self.conv1(x)
        x = x_shortcut = self.conv2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = x * x_shortcut  # Do NOT use in-place operation
        x = self.decoder(x)
        return x
