# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.random_projection import GaussianRandomProjection

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class GaussianRandomProjectionPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for dimensionality reduction based on Gaussian random projection algorithm.

    Method:
        The Gaussian random projection reduces the dimensionality by projecting the original input space on a randomly generated matrix where components are drawn from N(0, 1 / n_components).

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html

    Args:
        n_components: int
            Number of components to use.

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors(category="dimensionality_reduction").get("gauss_projection")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(
        self, random_state: int = 0, model: Any = None, n_components: int = 2
    ) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = GaussianRandomProjection(
            n_components=n_components, random_state=random_state
        )

    @staticmethod
    def name() -> str:
        return "gauss_projection"

    @staticmethod
    def subtype() -> str:
        return "dimensionality_reduction"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        cmin, cmax = base.PreprocessorPlugin.components_interval(*args, **kwargs)
        return [params.Integer("n_components", cmin, cmax)]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "GaussianRandomProjectionPlugin":
        self.model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "GaussianRandomProjectionPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = GaussianRandomProjectionPlugin
