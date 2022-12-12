# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.linear_model import LinearRegression

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.regression.base as base
from autoprognosis.utils.parallel import n_learner_jobs
import autoprognosis.utils.serialization as serialization


class LinearRegressionPlugin(base.RegressionPlugin):
    """Regression plugin based on the Linear Regression.

    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("linear_regression")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    solvers = ["auto", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]

    def __init__(self, model: Any = None, random_state: int = 0, **kwargs: Any) -> None:

        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        self.model = LinearRegression(
            n_jobs=n_learner_jobs(),
        )

    @staticmethod
    def name() -> str:
        return "linear_regression"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("max_iter", [100, 1000, 10000]),
            params.Integer("solver", 0, len(LinearRegressionPlugin.solvers) - 1),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "LinearRegressionPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "LinearRegressionPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = LinearRegressionPlugin
