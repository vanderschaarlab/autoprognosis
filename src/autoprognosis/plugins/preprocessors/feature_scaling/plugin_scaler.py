# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.preprocessing import StandardScaler

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class ScalerPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for feature scaling based on StandardScaler implementation.

    Method:
        The Scaler plugin standardizes the features by removing the mean and scaling to unit variance.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("scaler")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(self, random_state: int = 0, model: Any = None) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = StandardScaler()

    @staticmethod
    def name() -> str:
        return "scaler"

    @staticmethod
    def subtype() -> str:
        return "feature_scaling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "ScalerPlugin":
        self.model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "ScalerPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = ScalerPlugin
