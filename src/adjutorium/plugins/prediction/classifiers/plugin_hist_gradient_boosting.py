# stdlib
from typing import Any, List

# third party
import pandas as pd

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.classifiers.base as base
from adjutorium.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import adjutorium.utils.serialization as serialization

from sklearn.experimental import (  # noqa: F401,E402, isort:skip
    enable_hist_gradient_boosting,
)


from sklearn.ensemble import HistGradientBoostingClassifier  # isort:skip


class HistGradientBoostingPlugin(base.ClassifierPlugin):
    """Classification plugin based on the Histogram-based Gradient Boosting Classification Tree.

    Method:
        This estimator is much faster than GradientBoostingClassifier for big datasets (n_samples >= 10 000).

    Args:
        learning_rate: float
            Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
        max_depth: int
            The maximum depth of the individual regression estimators.


    Example:
        >>> from adjutorium.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("hist_gradient_boosting")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        calibration: int = 0,
        model: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_depth=max_depth
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "hist_gradient_boosting"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("max_depth", 5, 10),
            params.Categorical("learning_rate", [10 ** -p for p in range(1, 5)]),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "HistGradientBoostingPlugin":
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
    def load(cls, buff: bytes) -> "HistGradientBoostingPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = HistGradientBoostingPlugin
