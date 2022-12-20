# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class NormalTransformPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for feature scaling based on quantile information.

    Method:
        This method transforms the features to follow a normal distribution. Therefore, for a given feature, this transformation tends to spread out the most frequent values.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors().get("normal_transform")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
                    0         1         2         3
        0   -0.701131  1.061219 -1.205040 -1.138208
        1   -1.154434 -0.084214 -1.205040 -1.138208
        2   -1.523968  0.443066 -1.674870 -1.138208
        3   -1.710095  0.229099 -0.836836 -1.138208
        4   -0.923581  1.222611 -1.205040 -1.138208
        ..        ...       ...       ...       ...
        145  1.017901 -0.084214  0.778555  1.523968
        146  0.509020 -1.297001  0.547708  0.813193
        147  0.778555 -0.084214  0.778555  0.949666
        148  0.378986  0.824957  0.869109  1.523968
        149  0.109568 -0.084214  0.669219  0.627699

        [150 rows x 4 columns]
    """

    def __init__(
        self, random_state: int = 0, n_quantiles: int = 100, model: Any = None
    ) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles,
            random_state=random_state,
        )

    @staticmethod
    def name() -> str:
        return "normal_transform"

    @staticmethod
    def subtype() -> str:
        return "feature_scaling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "NormalTransformPlugin":

        self.model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "NormalTransformPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = NormalTransformPlugin
