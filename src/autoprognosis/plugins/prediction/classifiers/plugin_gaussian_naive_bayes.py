# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.naive_bayes import GaussianNB

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import autoprognosis.utils.serialization as serialization


class GaussianNaiveBayesPlugin(base.ClassifierPlugin):
    """Classification plugin based on the Gaussian Naive Bayes algorithm for classification.

    Method:
        The plugin implements the Gaussian Naive Bayes algorithm for classification. The likelihood of the features is assumed to be Gaussian.

    Args:
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("gaussian_naive_bayes")
        >>> plugin.fit_predict(...)
    """

    def __init__(
        self,
        calibration: int = 0,
        random_state: int = 0,
        model: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = GaussianNB()
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "gaussian_naive_bayes"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "GaussianNaiveBayesPlugin":
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
    def load(cls, buff: bytes) -> "GaussianNaiveBayesPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = GaussianNaiveBayesPlugin
