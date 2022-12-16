# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import autoprognosis.utils.serialization as serialization


class MultinomialNaiveBayesPlugin(base.ClassifierPlugin):
    """Classification plugin based on the Multinomial Naive Bayes algorithm.

    Method:
        The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).

    Args:
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed



    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("multinomial_naive_bayes")
        >>> plugin.fit_predict(...)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        calibration: int = 0,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = MultinomialNB(alpha=alpha)
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "multinomial_naive_bayes"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("alpha", 0.005, 5),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "MultinomialNaiveBayesPlugin":
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
    def load(cls, buff: bytes) -> "MultinomialNaiveBayesPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = MultinomialNaiveBayesPlugin
