# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.linear_model import LinearRegression

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.regression.base as base
import adjutorium.utils.serialization as serialization


class LinearRegressionPlugin(base.RegressionPlugin):
    """Regression plugin based on the Linear Regression.

    Example:
        >>> from adjutorium.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("linear_regression")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(self, model: Any = None, random_state: int = 0, **kwargs: Any) -> None:

        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        self.model = LinearRegression(
            n_jobs=-1,
        )

    @staticmethod
    def name() -> str:
        return "linear_regression"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

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
