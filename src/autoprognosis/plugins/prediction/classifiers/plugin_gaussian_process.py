# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
from autoprognosis.utils.parallel import n_learner_jobs
import autoprognosis.utils.serialization as serialization


class GaussianProcessPlugin(base.ClassifierPlugin):
    """Classification plugin based on Gaussian processes.

    Method:
        The plugin uses GaussianProcessClassifier, which implements Gaussian processes for classification purposes, more specifically for probabilistic classification, where test predictions take the form of class probabilities.

    Args:
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("gaussian_process")
        >>> plugin.fit_predict(...)
    """

    def __init__(self, calibration: int = 0, model: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = GaussianProcessClassifier(n_jobs=n_learner_jobs())
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
