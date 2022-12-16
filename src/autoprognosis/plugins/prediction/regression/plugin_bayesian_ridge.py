# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from sklearn.linear_model import BayesianRidge

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.regression.base as base
import autoprognosis.utils.serialization as serialization


class BayesianRidgePlugin(base.RegressionPlugin):
    """Bayesian ridge regression.

    Args:
        n_iter: int
            Maximum number of iterations. Should be greater than or equal to 1.
        tol: float
            Stop the algorithm if w has converged.
        random_state: int
            Random seed

    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("bayesian_ridge")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    def __init__(
        self,
        n_iter: int = 1000,
        tol: float = 0.001,
        hyperparam_search_iterations: Optional[int] = None,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:

        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        if hyperparam_search_iterations:
            n_iter = hyperparam_search_iterations

        self.model = BayesianRidge(
            n_iter=n_iter,
            tol=tol,
        )

    @staticmethod
    def name() -> str:
        return "bayesian_ridge"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("tol", [1e-3, 1e-2, 1e-4]),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "BayesianRidgePlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "BayesianRidgePlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = BayesianRidgePlugin
