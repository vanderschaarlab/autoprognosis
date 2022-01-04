# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.svm import LinearSVC

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.classifiers.base as base
from adjutorium.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import adjutorium.utils.serialization as serialization


class LinearSVMPlugin(base.ClassifierPlugin):
    """Classification plugin based on the Linear Support Vector Classification algorithm.

    Method:
        The plugin is based on LinearSVC, an implementation of Support Vector Classification for the case of a linear kernel.

    Args:
        penalty: str
            Specifies the norm used in the penalization.

    Example:
        >>> from adjutorium.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("linear_svm")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    penalties = ["l1", "l2"]

    def __init__(
        self, penalty: int = 1, calibration: int = 0, model: Any = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = LinearSVC(
            penalty=LinearSVMPlugin.penalties[penalty], dual=False, max_iter=10000
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
