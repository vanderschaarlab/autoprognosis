# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.tree import ExtraTreeClassifier

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.classifiers.base as base
from adjutorium.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import adjutorium.utils.serialization as serialization


class ExtraTreeClassifierPlugin(base.ClassifierPlugin):
    """Classification plugin based on extra-trees classifier.

    Method:
         The Extra-Trees classifierimplements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

    Args:
        criterion: str
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.

    Example:
        >>> from adjutorium.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("extra_tree_classifier")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    criterions = ["gini", "entropy"]

    def __init__(
        self, criterion: int = 0, calibration: int = 0, model: Any = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = ExtraTreeClassifier(
            criterion=ExtraTreeClassifierPlugin.criterions[criterion], max_depth=6
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "extra_tree_classifier"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer(
                "criterion", 0, len(ExtraTreeClassifierPlugin.criterions) - 1
            ),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "ExtraTreeClassifierPlugin":
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
    def load(cls, buff: bytes) -> "ExtraTreeClassifierPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = ExtraTreeClassifierPlugin
