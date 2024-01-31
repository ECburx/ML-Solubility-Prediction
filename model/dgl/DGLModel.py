from typing import Any

import numpy
import dgl
import torch
from model.abstractmodel import AbstractModel
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from tqdm import tqdm
from dgl import batch
from dgl.init import zero_initializer
from sklearn.metrics import mean_squared_error, r2_score


def _batcher(X, y, batch_size):
    split_fn = lambda arr: numpy.split(arr, numpy.arange(batch_size, len(arr), batch_size))
    return zip(split_fn(X), split_fn(y))


class DGLModel(AbstractModel):
    def __init__(
            self,
            model: torch.nn.Module,
            loss=SmoothL1Loss(reduction="none"),
            device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            lr: float = 0.01,
            weight_decay: float = 0,
            batch_size: int = 128,
    ):
        print(f"[DEVICE] {device}")
        self.device = device
        self.batch_size = batch_size
        self.loss = loss
        self.optim = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scores = None
        super().__init__(model=model.to(self.device))

    def fit(
            self,
            X, y,
            *args,
            epochs: int = 100,
            early_stop_epochs: int = None,
            min_epochs: int = 100,
            val_X=None, val_y=None,
            verbose=True,
            **kwargs
    ) -> dict:
        self.scores = {"trn_loss": [], "val_loss": [], "rmse": [], "r2": []}
        best_scores = {"val_loss": float('inf'), "rmse": float('inf'), "r2": float('inf')}

        patience = early_stop_epochs
        for e in (bar := tqdm(range(epochs))) if verbose else range(epochs):
            trn_loss = self.train_epoch(X, y)
            self.scores["trn_loss"].append(trn_loss)

            if val_X is None or val_y is None:
                continue

            val_loss, rmse, r2 = self.validation_epoch(val_X, val_y)
            self.scores["val_loss"].append(val_loss)
            self.scores["rmse"].append(rmse)
            self.scores["r2"].append(r2)

            if verbose:
                bar.set_postfix_str(
                    f"trn_loss:{trn_loss:.3f} val_loss:{val_loss:.3f} rmse:{rmse:.3f} r2:{r2:.3f}")

            if e < min_epochs:
                continue

            if early_stop_epochs is not None:
                if rmse < best_scores["rmse"]:
                    best_scores["rmse"] = rmse
                    patience = early_stop_epochs
                elif r2 < best_scores["r2"]:
                    best_scores["r2"] = r2
                    patience = early_stop_epochs
                elif val_loss < best_scores["val_loss"]:
                    best_scores["val_loss"] = val_loss
                    patience = early_stop_epochs
                else:
                    patience -= 1
                if patience <= 0:
                    break

        return self.scores

    def predict(self, X, **kwargs) -> Any:
        self.model.eval()
        with torch.no_grad():
            graphs, _ = self.collate(X)
            return self._predict(graphs)

    def _predict(self, graphs) -> Any:
        return self.model(graphs, graphs.ndata.pop('h').to(self.device))

    def train_epoch(self, trn_X, trn_y):
        sum_loss = 0
        for X, y in _batcher(trn_X, trn_y, self.batch_size):
            X, y = self.collate(X, y)
            self.model.train()
            pred = self._predict(X)
            loss = self.loss(pred, y).mean()
            self.optim.zero_grad()
            loss.backward()
            sum_loss += loss
            self.optim.step()
        return sum_loss

    def validation_epoch(self, val_X, val_y):
        self.model.eval()
        with torch.no_grad():
            graphs, y = self.collate(val_X, val_y)
            pred = self._predict(graphs)
            y_true, y_pred = y.cpu(), pred.cpu()
            return (self.loss(y_pred, y_true).mean(),
                    mean_squared_error(y_true, y_pred, squared=False),
                    r2_score(y_true, y_pred))

    def collate(self, X, y=None) -> tuple[dgl.DGLGraph, torch.Tensor]:
        graphs = batch(X.tolist())
        graphs.set_n_initializer(zero_initializer)
        graphs.set_e_initializer(zero_initializer)
        graphs = graphs.to(self.device)

        if y is None:
            y = None
        else:
            y = torch.FloatTensor(y.to_list()).to(self.device).unsqueeze(1)
            # else:
            #     y = torch.FloatTensor(trn_y.values).to(self.device)

        return graphs, y
