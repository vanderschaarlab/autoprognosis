# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class VarianceThresholdPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for dimensionality reduction based on removing features with low variance.

    Method:
        VarianceThreshold is a simple baseline approach to feature selection. It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features, i.e. features that have the same value in all samples.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html

    Args:
        threshold: float
            Features with a training-set variance lower than this threshold will be removed.

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("variance_threshold", threshold=1.0)
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
               0
        0    1.4
        1    1.4
        2    1.3
        3    1.5
        4    1.4
        ..   ...
        145  5.2
        146  5.0
        147  5.2
        148  5.4
        149  5.1

        [150 rows x 1 columns]
    """

    def __init__(
        self, random_state: int = 0, model: Any = None, threshold: float = 0.001
    ) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = VarianceThreshold(threshold=threshold)

    @staticmethod
    def name() -> str:
        return "variance_threshold"

    @staticmethod
    def subtype() -> str:
        return "dimensionality_reduction"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "VarianceThresholdPlugin":
        self.model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.model.get_support()]

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "VarianceThresholdPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = VarianceThresholdPlugin
