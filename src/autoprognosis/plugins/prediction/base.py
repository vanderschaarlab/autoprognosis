# stdlib
from abc import abstractmethod
from typing import Any

# third party
import pandas as pd

# autoprognosis absolute
import autoprognosis.logger as log
import autoprognosis.plugins.core.base_plugin as plugin


class PredictionPlugin(plugin.Plugin):
    """Base class for the prediction plugins.

    It provides the implementation for plugin.Plugin.type() static method.

    Each derived class must implement the following methods(inherited from plugin.Plugin):
        name() - a static method that returns the name of the plugin.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `params.Params` derived objects.
        _fit() - internal implementation, called by the `fit` method.
        _predict() - internal implementation, called by the `predict` method.
        _predict_proba() - internal implementation, called by the `predict_proba` method.

    If any method implementation is missing, the class constructor will fail.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def type() -> str:
        return "prediction"

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "Prediction plugins do not implement the 'transform' method"
        )

    def score(self, X: pd.DataFrame, y: pd.DataFrame, metric: str = "aucroc") -> float:
        raise NotImplementedError(f"Score not implemented for {self.name()}")

    def explain(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError(f"Explainer not implemented for {self.name()}")

    def predict_proba(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if not self.is_fitted():
            raise RuntimeError("Fit the model first")

        log.debug(f"Predicting using {self.fqdn()}, input shape = {X.shape}")
        X = self._preprocess_inference_data(X)
        result = pd.DataFrame(self._predict_proba(X, *args, **kwargs))

        return result

    @abstractmethod
    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        ...
