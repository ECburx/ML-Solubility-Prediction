from abc import ABC, abstractmethod


class AbstractModel(ABC):
    def __init__(self, model=None):
        self.model = model

    @abstractmethod
    def fit(self, trn_X, trn_y, *args, val_X=None, val_y=None, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, **kwargs):
        pass
