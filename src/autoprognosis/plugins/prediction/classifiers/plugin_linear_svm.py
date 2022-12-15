# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.svm import LinearSVC

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import autoprognosis.utils.serialization as serialization


class LinearSVMPlugin(base.ClassifierPlugin):
    """Classification plugin based on the Linear Support Vector Classification algorithm.

    Method:
        The plugin is based on LinearSVC, an implementation of Support Vector Classification for the case of a linear kernel.

    Args:
        penalty: int
            Specifies the norm used in the penalization. 0: l1, 1: l2
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("linear_svm")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    penalties = ["l1", "l2"]

    def __init__(
        self,
        penalty: int = 1,
        calibration: int = 0,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = LinearSVC(
            penalty=LinearSVMPlugin.penalties[penalty],
            dual=False,
            max_iter=10000,
            random_state=random_state,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "linear_svm"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("penalty", 0, len(LinearSVMPlugin.penalties) - 1),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "LinearSVMPlugin":
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
    def load(cls, buff: bytes) -> "LinearSVMPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = LinearSVMPlugin
