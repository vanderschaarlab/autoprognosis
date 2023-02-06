# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import autoprognosis.utils.serialization as serialization


class GradientBoostingPlugin(base.ClassifierPlugin):
    """Classification plugin based on the Gradient boosting method.

    Method:
        Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. When a decision tree is the weak learner, the resulting algorithm is called gradient boosted trees, which usually outperforms random forest.

    Args:
        n_estimators: int
            The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        learning_rate: float
            Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
        max_depth: int
            The maximum depth of the individual regression estimators.
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("gradient_boosting")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        calibration: int = 0,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "gradient_boosting"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("n_estimators", 10, 500, 10),
            params.Integer("max_depth", 1, 8),
            params.Categorical("learning_rate", [10**-p for p in range(1, 5)]),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "GradientBoostingPlugin":
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
    def load(cls, buff: bytes) -> "GradientBoostingPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = GradientBoostingPlugin
