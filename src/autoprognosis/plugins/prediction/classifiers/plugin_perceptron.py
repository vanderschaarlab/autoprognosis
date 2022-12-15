# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.linear_model import Perceptron

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import autoprognosis.utils.serialization as serialization


class PerceptronPlugin(base.ClassifierPlugin):
    """Classification plugin based on perceptrons.

    Method:
         Perceptron is simple classification algorithm suitable for large scale learning. By default, it does not require a learning rate and it updates its model only on mistakes.

    Args:
        penalty: str
            The penalty to be used: {‘l2’,’l1’,’elasticnet’}
        alpha: float
            Constant that multiplies the regularization term if regularization is used.
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed




    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("perceptron")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    penalties = ["l1", "l2", "elasticnet"]

    def __init__(
        self,
        penalty: int = 1,
        alpha: float = 0.0001,
        calibration: int = 0,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = Perceptron(
            penalty=PerceptronPlugin.penalties[penalty],
            alpha=alpha,
            random_state=random_state,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "perceptron"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("alpha", 0.00005, 0.001),
            params.Integer("penalty", 0, len(PerceptronPlugin.penalties) - 1),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "PerceptronPlugin":
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
    def load(cls, buff: bytes) -> "PerceptronPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = PerceptronPlugin
