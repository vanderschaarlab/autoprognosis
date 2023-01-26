# stdlib
from typing import Any

# third party
import pandas as pd

# autoprognosis absolute
import autoprognosis.logger as log
import autoprognosis.plugins.core.base_plugin as plugin
import autoprognosis.plugins.prediction.base as prediction_base
import autoprognosis.plugins.utils.cast as cast
from autoprognosis.utils.tester import classifier_metrics


class ClassifierPlugin(prediction_base.PredictionPlugin):
    """Base class for the classifier plugins.

    It provides the implementation for plugin.Plugin's subtype, _fit and _predict methods.

    Each derived class must implement the following methods(inherited from plugin.Plugin):
        name() - a static method that returns the name of the plugin.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `Params` derived objects.

    If any method implementation is missing, the class constructor will fail.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.args = kwargs

        super().__init__()

    @staticmethod
    def subtype() -> str:
        return "classifier"

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> plugin.Plugin:
        X = self._preprocess_training_data(X)

        log.debug(f"Training using {self.fqdn()}, input shape = {X.shape}")
        if len(args) == 0:
            raise RuntimeError("Training requires X, y")
        Y = cast.to_dataframe(args[0]).values.ravel()

        self._fit(X, Y, **kwargs)

        self._fitted = True
        log.debug(f"Done training using {self.fqdn()}, input shape = {X.shape}")

        return self

    def score(self, X: pd.DataFrame, y: pd.DataFrame, metric: str = "aucroc") -> float:
        ev = classifier_metrics()

        preds = self.predict_proba(X)
        return ev.score_proba(y, preds)[metric]

    def get_args(self) -> dict:
        return self.args
