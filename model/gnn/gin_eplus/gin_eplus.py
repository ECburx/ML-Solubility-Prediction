"""
Derived from:
    https://github.com/RBrossard/GINEPLUS/blob/master

Reference:
    Brossard, R., Frigo, O., & Dehaene, D. (2020).
    Graph convolutions that can finally model local structure. arXiv.
    https://doi.org/10.48550/arXiv.2011.15069
"""

import pandas as pd
import torch

from data.dataset import Dataset
from model.abstractmodel import AbstractModel

from model.gnn.gin_eplus.util import GIN_Classifier_Model


class GIN_EPLUS(AbstractModel):
    def __init__(
            self,
            task_type: str,
            out_dims: int,
            max_epochs: int = 100,
            hidden_dims: int = 100,
            hidden_lyrs: int = 3,
            dropout: float = 0.5,
            virtual_node: bool = False,
            conv_radius: int = 4,
            conv_type: str = "gin+",
            batch_size: int = 128,
            num_workers: int = 0,
            lr: float = None
    ):
        """

        :param task_type:
        :param out_dims:
        :param epochs:
        :param hidden_dims: embedding dimension of the nodes. (default:100)
        :param hidden_lyrs: number of convolution layers. (default: 3)
        :param dropout: dropout rate (default 0.5)
        :param virtual_node: If specified, uses a virtual node.
        :param conv_radius: Radius of the GINE+ and NaiveGINE+ convolutions. (default: 4)
        :param conv_type: Type of convolution, must be one of gin+, naivegin+, gin or gcn. (default: gin+)
        """
        super().__init__(task_type)

        self.batch_size = batch_size
        self.num_workers = num_workers

        from pytorch_lightning import Trainer
        self.trainer = Trainer(
                accelerator="gpu",
                gpus=[0],
                auto_select_gpus=True,
                max_epochs=max_epochs,
                auto_lr_find=True if lr is None else False,
                num_sanity_val_steps=0
        )

        if task_type == "classification":
            self.model = GIN_Classifier_Model(
                    out_dims=out_dims,
                    hidden_dims=hidden_dims,
                    hidden_lyrs=hidden_lyrs,
                    dropout=dropout,
                    virtual_node=virtual_node,
                    conv_radius=conv_radius,
                    conv_type=conv_type,
                    lr=lr
            )
        else:
            assert out_dims == 1
            pass  # TODO

    def fit(
            self,
            dataset: Dataset,
            *args,
            eval_dataset: Dataset = None,
            graph_col_name: str = "graph",
            **kwargs
    ):
        self.model.train()

        from torch_geometric.data import DataLoader
        trn_loader = DataLoader(
                dataset.X[graph_col_name].tolist(),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
        )
        eval_loader = DataLoader(
                eval_dataset.X[graph_col_name].tolist(),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
        ) if eval_dataset is not None else None

        self.trainer.fit(
                model=self.model,
                train_dataloaders=trn_loader,
                val_dataloaders=eval_loader
        )

    def save(self, path):
        pass

    def load(self, path):
        pass

    def predict(self, dataset: Dataset, graph_col_name: str = "graph"):
        self.model.eval()

        from torch_geometric.data import DataLoader
        loader = DataLoader(
                dataset.X[graph_col_name].tolist(),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
        )

        prediction = self.trainer.predict(self.model, loader)

        if self.task_type == "classification":
            _, prediction = torch.max(torch.cat(prediction, dim=0), 1)
            prediction = prediction.cpu().numpy()
        else:
            pass  # TODO
        return pd.DataFrame(prediction)
