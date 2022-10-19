# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class UniformTransformPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for feature scaling based on quantile information.

    Method:
        This method transforms the features to follow a uniform distribution. Therefore, for a given feature, this transformation tends to spread out the most frequent values.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("uniform_transform")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
                    0         1         2         3
        0    0.241611  0.855705  0.114094  0.127517
        1    0.124161  0.466443  0.114094  0.127517
        2    0.063758  0.671141  0.046980  0.127517
        3    0.043624  0.590604  0.201342  0.127517
        4    0.177852  0.889262  0.114094  0.127517
        ..        ...       ...       ...       ...
        145  0.845638  0.466443  0.781879  0.936242
        146  0.694631  0.097315  0.708054  0.791946
        147  0.781879  0.466443  0.781879  0.828859
        148  0.647651  0.795302  0.807606  0.936242
        149  0.543624  0.466443  0.748322  0.734899

        [150 rows x 4 columns]
    """

    def __init__(self, random_state: int = 0, model: Any = None) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = QuantileTransformer(n_quantiles=30)

    @staticmethod
    def name() -> str:
        return "uniform_transform"

    @staticmethod
    def subtype() -> str:
        return "feature_scaling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "UniformTransformPlugin":
        self.model.fit(X)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "UniformTransformPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = UniformTransformPlugin
