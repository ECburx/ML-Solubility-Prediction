from data.dataset import Dataset
from model.abstractmodel import AbstractModel
import torch
import torch_geometric


class SPGNN(AbstractModel):
    """
    Derived from and DEPENDS ON olorenchemengine v.1.0.9

    Reference:
        Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec, J. (2019). Strategies for Pre-training
        Graph Neural Networks. arXiv.
        https://doi.org/10.48550/arXiv.1905.12265
    """

    AVAILABLE_MODEL_TYPE = [
            "contextpred",
            "edgepred",
            "infomax",
            "masking",
            "supervised_contextpred",
            "supervised_edgepred",
            "supervised_infomax",
            "supervised_masking",
            "supervised",
            "gat_supervised_contextpred",
            "gat_supervised",
            "gat_contextpred",
    ]

    def __init__(
            self,
            task_type: str,
            num_tasks: int,
            model_type: str = "contextpred",
            batch_size: int = 128,
            epochs: int = 100,
            lr=0.001,
            lr_scale=1,
            decay=0,
            num_layer: int = 5,
            emb_dim: int = 300,
            dropout_ratio: float = 0.5,
            graph_pooling: str = "mean",
            JK: str = "last",
            gnn_type: str = "gin",
            num_workers: int = 0,
            map_location: str = "cpu",
    ):
        super().__init__(task_type, description_info="SPGNN_Graph")

        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.map_location = map_location
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if "gat" in model_type:
            gnn_type = "gat"
            emb_dim = 300

        if task_type == "regression":
            assert num_tasks == 1

        from olorenchemengine.external.SPGNN.main import GNN_graphpred
        self.model = GNN_graphpred(
                num_layer=num_layer,
                emb_dim=emb_dim,
                num_tasks=num_tasks,
                JK=JK,
                drop_ratio=dropout_ratio,
                graph_pooling=graph_pooling,
                gnn_type=gnn_type,
        ).to(self.device)

        # from olorenchemengine.internal import download_public_file
        # input_model_file = download_public_file(f"SPGNN_saves/{self.model_type}.pth")
        #
        # self.model.from_pretrained(input_model_file, map_location=self.map_location)

        model_param_group = [{"params": self.model.gnn.parameters()}]
        if graph_pooling == "attention":
            model_param_group.append(
                    {"params": self.model.pool.parameters(), "lr": lr * lr_scale}
            )
        model_param_group.append(
                {"params": self.model.graph_pred_linear.parameters(), "lr": lr * lr_scale}
        )

        self.optimizer = torch.optim.Adam(model_param_group, lr=lr, weight_decay=decay)
        self.sigmoid = torch.nn.Sigmoid()

    def fit(
            self,
            dataset: Dataset,
            *args,
            graph_col_name: str = "graph",
            criterion_weight: torch.Tensor = None,
            **kwargs
    ):
        loader = torch_geometric.data.DataLoader(
                dataset.X[graph_col_name].tolist(),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
        )

        if criterion_weight is not None:
            criterion_weight = criterion_weight.to(self.device)

        if self.task_type == "classification":
            criterion = torch.nn.CrossEntropyLoss(weight=criterion_weight)
        else:
            criterion = torch.nn.MSELoss()

        from tqdm import tqdm
        for epoch in (bar := tqdm(range(self.epochs))):
            self.model.train()
            loss_sum = 0

            for batch_idx, batch in enumerate(loader):
                batch = batch.to(self.device)
                pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                true = batch.y.to(torch.long if self.task_type == "classification" else torch.float)

                loss_matrix = criterion(pred, true)
                self.optimizer.zero_grad()
                loss = torch.mean(loss_matrix)
                loss_sum += loss.item()
                loss.backward()
                self.optimizer.step()

            bar.set_postfix({"loss": loss_sum})

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(
            self,
            dataset: Dataset,
            graph_col_name: str = "graph",
            probabilities: bool = False
    ):
        loader = torch_geometric.data.DataLoader(
                dataset.X[graph_col_name].tolist(),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
        )

        self.model.eval()
        prediction = []

        for batch_idx, batch in enumerate(loader):
            batch = batch.to(self.device)

            with torch.no_grad():
                pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            prediction.append(pred)

        if self.task_type == "classification":
            if probabilities:
                prediction = torch.cat(prediction, dim=0).cpu().numpy()
            else:
                _, prediction = torch.max(torch.cat(prediction, dim=0), 1)
                prediction = prediction.cpu().numpy()
        else:
            prediction = torch.cat(prediction, dim=0).cpu().numpy()
        return prediction
