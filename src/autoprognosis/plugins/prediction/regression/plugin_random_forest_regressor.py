# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.regression.base as base
from autoprognosis.utils.parallel import n_learner_jobs
import autoprognosis.utils.serialization as serialization


class RandomForestRegressionPlugin(base.RegressionPlugin):
    """Regression plugin based on Random forests.

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
        random_state: int
            Random seed

    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("random_forest")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    criterions = ["squared_error", "absolute_error", "friedman_mse", "poisson"]

    def __init__(
        self,
        n_estimators: int = 50,
        criterion: int = 0,
        min_samples_split: int = 2,
        bootstrap: bool = True,
        min_samples_leaf: int = 2,
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

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion=RandomForestRegressionPlugin.criterions[criterion],
            min_samples_split=min_samples_split,
            max_depth=4,
            bootstrap=bootstrap,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_learner_jobs(),
            random_state=random_state,
        )

    @staticmethod
    def name() -> str:
        return "random_forest_regressor"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer(
                "criterion", 0, len(RandomForestRegressionPlugin.criterions) - 1
            ),
            params.Categorical("min_samples_split", [2, 5, 10]),
            params.Categorical("min_samples_leaf", [2, 5, 10]),
            params.Integer("n_estimators", 10, 10000),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "RandomForestRegressionPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "RandomForestRegressionPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = RandomForestRegressionPlugin
