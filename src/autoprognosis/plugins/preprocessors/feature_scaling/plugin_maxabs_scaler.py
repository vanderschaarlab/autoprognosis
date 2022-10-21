# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class MaxAbsScalerPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for feature scaling based on maximum absolute value.

    Method:
        The MaxAbs estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set will be 1.0.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("maxabs_scaler")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
                    0         1         2     3
        0    0.645570  0.795455  0.202899  0.08
        1    0.620253  0.681818  0.202899  0.08
        2    0.594937  0.727273  0.188406  0.08
        3    0.582278  0.704545  0.217391  0.08
        4    0.632911  0.818182  0.202899  0.08
        ..        ...       ...       ...   ...
        145  0.848101  0.681818  0.753623  0.92
        146  0.797468  0.568182  0.724638  0.76
        147  0.822785  0.681818  0.753623  0.80
        148  0.784810  0.772727  0.782609  0.92
        149  0.746835  0.681818  0.739130  0.72

        [150 rows x 4 columns]
    """

    def __init__(self, random_state: int = 0, model: Any = None) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = MaxAbsScaler()

    @staticmethod
    def name() -> str:
        return "maxabs_scaler"

    @staticmethod
    def subtype() -> str:
        return "feature_scaling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "MaxAbsScalerPlugin":
        self.model.fit(X)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "MaxAbsScalerPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = MaxAbsScalerPlugin
