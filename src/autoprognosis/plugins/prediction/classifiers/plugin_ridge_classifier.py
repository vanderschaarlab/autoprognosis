# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.linear_model import RidgeClassifier

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import autoprognosis.utils.serialization as serialization


class RidgeClassifierPlugin(base.ClassifierPlugin):
    """Classification plugin based on the Ridge classifier.

    Method:
        The RidgeClassifier converts the target values into {-1, 1} and then treats the problem as a regression task (multi-output regression in the multiclass case).

    Args:
        solver: str
            Algorithm to use in the optimization problem: {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed



    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("ridge_classifier")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg"]  # , "sag", "saga"]

    def __init__(
        self,
        solver: int = 0,
        calibration: int = 0,
        random_state: int = 0,
        model: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = RidgeClassifier(
            solver=RidgeClassifierPlugin.solvers[solver], random_state=random_state
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "ridge_classifier"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer(
                "solver",
                0,
                len(RidgeClassifierPlugin.solvers) - 1,
            ),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "RidgeClassifierPlugin":
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
    def load(cls, buff: bytes) -> "RidgeClassifierPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = RidgeClassifierPlugin
