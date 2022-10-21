# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.preprocessing import Normalizer

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class FeatureNormalizerPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for sample normalization based on L2 normalization.

    Method:
        Normalization is the process of scaling individual samples to have unit norm.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("feature_normalizer")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
                    0         1         2         3
        0    0.803773  0.551609  0.220644  0.031521
        1    0.828133  0.507020  0.236609  0.033801
        2    0.805333  0.548312  0.222752  0.034269
        3    0.800030  0.539151  0.260879  0.034784
        4    0.790965  0.569495  0.221470  0.031639
        ..        ...       ...       ...       ...
        145  0.721557  0.323085  0.560015  0.247699
        146  0.729654  0.289545  0.579090  0.220054
        147  0.716539  0.330710  0.573231  0.220474
        148  0.674671  0.369981  0.587616  0.250281
        149  0.690259  0.350979  0.596665  0.210588

        [150 rows x 4 columns]
    """

    def __init__(self, random_state: int = 0, model: Any = None) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = Normalizer()

    @staticmethod
    def name() -> str:
        return "feature_normalizer"

    @staticmethod
    def subtype() -> str:
        return "feature_scaling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "FeatureNormalizerPlugin":
        self.model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "FeatureNormalizerPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = FeatureNormalizerPlugin
