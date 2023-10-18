from typing import Union

from data.dataset import Dataset
from model.abstractmodel import AbstractModel


class TabNet(AbstractModel):
    def __init__(self,
                 task_type: str,
                 num_tasks: int = None,
                 n_d: int = 8,
                 n_a: int = 8,
                 n_steps: int = 3,
                 n_independent: int = 2,
                 n_shared: int = 2,
                 gamma: float = 1.3,
                 cat_idxs: list[int] = None,
                 cat_dims: list[int] = None,
                 momentum: float = 0.02,
                 seed: int = 0,
                 verbose: int = 1,
                 device: str = "auto",
                 mask_type: str = "sparsemax",
                 **kwargs):
        """

        :param task_type:
        :param num_tasks:
        :param n_d: Width of the decision prediction layer. Suggestion: [8, 64]
        :param n_a: Width of the attention embedding for each mask. Suggestion: [n_d = n_a]
        :param n_steps: Number of steps in the architecture. Suggestion: [3, 10]
        :param n_independent: Number of independent Gated Linear Units layers at each step. Suggestion [1, 5]
        :param n_shared: Number of shared Gated Linear Units at each step. Suggestion [1, 5]
        :param gamma: A value close to 1 will make mask selection least correlated between layers. Range From [1.0, 2.0]
        :param cat_idxs: List of categorical features indices.
        :param cat_dims: List of categorical features number of modalities.
        :param momentum: Momentum for batch normalization. Suggestion [0.01, 0.4]
        :param seed: Random Seed for reproducibility.
        :param verbose: Verbosity. Set to 1 to see every epoch, 0 to get None.
        :param device: [cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, ort, mps, xla, lazy, vulkan, meta,
                        hpu, privateuseone]
        :param mask_type: Masking function to use for selecting features. ["sparsemax", "entmax"]
        :param kwargs:
        """
        super().__init__(task_type, "Any Tabular Descriptor")

        if self.task_type == "classification":
            assert num_tasks is not None
            if num_tasks > 2:
                from pytorch_tabnet.tab_model import TabNetClassifier
                self.model = TabNetClassifier(
                        n_d=n_d,
                        n_a=n_a,
                        n_steps=n_steps,
                        n_independent=n_independent,
                        n_shared=n_shared,
                        gamma=gamma,
                        cat_idxs=[] if cat_idxs is None else cat_idxs,
                        cat_dims=[] if cat_dims is None else cat_dims,
                        momentum=momentum,
                        seed=seed,
                        verbose=verbose,
                        device_name=device,
                        mask_type=mask_type,
                        **kwargs
                )
            else:
                pass  # TODO
        else:
            pass  # TODO

    def fit(self,
            dataset: Dataset,
            *args,
            loss_fn=None,
            eval_dataset: Dataset = None,
            eval_metric: list[str] = None,
            max_epochs: int = 100,
            patience: int = 10,
            weights: Union[int, dict] = 0,
            **kwargs
            ):
        """
        :param loss_fn:
        :param dataset:
        :param args:
        :param eval_dataset:
        :param eval_metric: https://dreamquark-ai.github.io/tabnet/generated_docs/README.html#default-eval-metric
        :param max_epochs: Maximum number of epochs for training.
        :param patience: Early stopping patience.
        :param weights: Only for Classifier. [1: automated sampling with inverse class occurrences]
        :param kwargs:
        :return:
        """
        self.model.fit(
                X_train=dataset.X.values,
                y_train=dataset.y.values.reshape(-1),
                eval_set=[(eval_dataset.X.values, eval_dataset.y.values.reshape(-1))]
                if eval_dataset is not None else None,
                eval_metric=eval_metric,
                max_epochs=max_epochs,
                patience=patience,
                weights=weights,
                loss_fn=loss_fn,
                **kwargs
        )

    def save(self, path):
        pass

    def load(self, path):
        pass

    def predict(
            self,
            dataset: Dataset,
            probabilities: bool = False):
        if probabilities:
            return self.model.predict_proba(dataset.X.values)
        else:
            return self.model.predict(dataset.X.values)

    from pytorch_tabnet.metrics import Metric

    class Quadratic_Cohen_Kappa(Metric):
        def __init__(self):
            self._name = "Quadratic_Cohen_Kappa"
            self._maximize = True

        def __call__(self, y_true, y_pred):
            from data.metric import sklearn_quadratic_cohen_kappa
            return sklearn_quadratic_cohen_kappa(y_pred.argmax(axis=1), y_true)
