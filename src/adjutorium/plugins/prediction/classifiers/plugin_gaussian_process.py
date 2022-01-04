# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.classifiers.base as base
from adjutorium.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import adjutorium.utils.serialization as serialization


class GaussianProcessPlugin(base.ClassifierPlugin):
    """Classification plugin based on Gaussian processes.

    Method:
        The plugin uses GaussianProcessClassifier, which implements Gaussian processes for classification purposes, more specifically for probabilistic classification, where test predictions take the form of class probabilities.

    Example:
        >>> from adjutorium.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("gaussian_process")
        >>> plugin.fit_predict(...)
    """

    def __init__(self, calibration: int = 0, model: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = GaussianProcessClassifier(n_jobs=-1)
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "gaussian_process"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "GaussianProcessPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "GaussianProcessPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = GaussianProcessPlugin
