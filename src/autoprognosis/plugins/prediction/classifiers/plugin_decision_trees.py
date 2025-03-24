# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
import autoprognosis.utils.serialization as serialization
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)


class DecisionTreePlugin(base.ClassifierPlugin):
    """Classification plugin based on the Decision trees.

    Method:
        Decision Trees are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.

    Args:
        criterion: int
            The function to measure the quality of a split. Supported criteria are “gini”(0) for the Gini impurity and “entropy”(1) for the information gain.
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("decision_trees")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    criterions = ["gini", "entropy"]

    def __init__(
        self,
        criterion: int = 0,
        calibration: int = 0,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = DecisionTreeClassifier(
            criterion=DecisionTreePlugin.criterions[criterion],
            random_state=random_state,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "decision_trees"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("criterion", 0, len(DecisionTreePlugin.criterions) - 1),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "DecisionTreePlugin":
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
    def load(cls, buff: bytes) -> "DecisionTreePlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = DecisionTreePlugin
