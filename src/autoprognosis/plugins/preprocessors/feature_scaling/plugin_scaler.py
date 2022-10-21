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
                    0         1         2         3
        0   -0.900681  1.019004 -1.340227 -1.315444
        1   -1.143017 -0.131979 -1.340227 -1.315444
        2   -1.385353  0.328414 -1.397064 -1.315444
        3   -1.506521  0.098217 -1.283389 -1.315444
        4   -1.021849  1.249201 -1.340227 -1.315444
        ..        ...       ...       ...       ...
        145  1.038005 -0.131979  0.819596  1.448832
        146  0.553333 -1.282963  0.705921  0.922303
        147  0.795669 -0.131979  0.819596  1.053935
        148  0.432165  0.788808  0.933271  1.448832
        149  0.068662 -0.131979  0.762758  0.790671

        [150 rows x 4 columns]

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
