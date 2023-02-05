# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
from autoprognosis.utils.parallel import n_learner_jobs
import autoprognosis.utils.serialization as serialization


class RandomForestPlugin(base.ClassifierPlugin):
    """Classification plugin based on Random forests.

    Method:
        A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

    Args:
        n_estimators: int
            The number of trees in the forest.
        criterion: str
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
        min_samples_split: int
            The minimum number of samples required to split an internal node.
        boostrap: bool
            Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
        min_samples_leaf: int
            The minimum number of samples required to be at a leaf node.
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed



    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("random_forest")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    criterions = ["gini", "entropy"]

    def __init__(
        self,
        n_estimators: int = 100,
        criterion: int = 0,
        min_samples_split: int = 2,
        bootstrap: bool = True,
        min_samples_leaf: int = 2,
        calibration: int = 0,
        max_depth: int = 4,
        model: Any = None,
        hyperparam_search_iterations: Optional[int] = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        if hyperparam_search_iterations:
            n_estimators = int(hyperparam_search_iterations)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=RandomForestPlugin.criterions[criterion],
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            bootstrap=bootstrap,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_learner_jobs(),
            random_state=random_state,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "random_forest"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("criterion", 0, len(RandomForestPlugin.criterions) - 1),
            params.Integer("n_estimators", 100, 10000),
            params.Integer("max_depth", 1, 7),
            params.Categorical("min_samples_split", [2, 5, 10]),
            params.Categorical("bootstrap", [True, False]),
            params.Categorical("min_samples_leaf", [2, 5, 10]),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "RandomForestPlugin":
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
    def load(cls, buff: bytes) -> "RandomForestPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = RandomForestPlugin
