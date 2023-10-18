from model.dgl.DGLModel import DGLModel
import torch


class NF(DGLModel):
    def __init__(
            self,
            n_tasks: int,
            in_feats: int,
            hidden_feats: list[int],
            max_degree: int = 10,
            activation: torch.nn.functional = torch.nn.functional.relu,
            batchnorm: bool = True,
            dropout: float = 0,
            predictor_hidden_size: int = 128,
            predictor_batchnorm: bool = True,
            predictor_dropout: float = 0,
            predictor_activation: torch.nn.functional = torch.tanh,
            **kwargs
    ):
        from dgllife.model import NFPredictor
        layer_num = len(hidden_feats)
        super().__init__(
                model=NFPredictor(
                        in_feats=in_feats,
                        n_tasks=n_tasks,
                        hidden_feats=hidden_feats,
                        max_degree=max_degree,
                        activation=[activation] * layer_num,
                        batchnorm=[batchnorm] * layer_num,
                        dropout=[dropout] * layer_num,
                        predictor_hidden_size=predictor_hidden_size,
                        predictor_batchnorm=predictor_batchnorm,
                        predictor_dropout=predictor_dropout,
                        predictor_activation=predictor_activation
                ),
                **kwargs
        )

    def reset_parameters(self):
        self.model.gnn.reset_parameters()
