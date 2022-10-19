# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class MinMaxScalerPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for feature scaling to a given range.

    Method:
        The MinMax estimator scales and translates each feature individually such that it is in the given range on the training set, e.g. between zero and one.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("minmax_scaler")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
                    0         1         2         3
        0    0.222222  0.625000  0.067797  0.041667
        1    0.166667  0.416667  0.067797  0.041667
        2    0.111111  0.500000  0.050847  0.041667
        3    0.083333  0.458333  0.084746  0.041667
        4    0.194444  0.666667  0.067797  0.041667
        ..        ...       ...       ...       ...
        145  0.666667  0.416667  0.711864  0.916667
        146  0.555556  0.208333  0.677966  0.750000
        147  0.611111  0.416667  0.711864  0.791667
        148  0.527778  0.583333  0.745763  0.916667
        149  0.444444  0.416667  0.694915  0.708333

        [150 rows x 4 columns]

    """

    def __init__(self, random_state: int = 0, model: Any = None) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = MinMaxScaler()

    @staticmethod
    def name() -> str:
        return "minmax_scaler"

    @staticmethod
    def subtype() -> str:
        return "feature_scaling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "MinMaxScalerPlugin":
        self.model.fit(X)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "MinMaxScalerPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = MinMaxScalerPlugin
