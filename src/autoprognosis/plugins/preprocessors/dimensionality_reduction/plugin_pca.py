# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from sklearn.decomposition import PCA

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class PCAPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for dimensionality reduction based on the PCA method.

    Method:
        PCA is used to decompose a multivariate dataset in a set of successive orthogonal components that explain a maximum amount of the variance.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Args:
        n_components: int
            Number of components to use.

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors(category="dimensionality_reduction").get("pca")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(
        self, random_state: int = 0, model: Any = None, n_components: int = 2
    ) -> None:
        super().__init__()
        self.random_state = random_state
        self.n_components = n_components
        self.model: Optional[PCA] = None

        if model:
            self.model = model

    @staticmethod
    def name() -> str:
        return "pca"

    @staticmethod
    def subtype() -> str:
        return "dimensionality_reduction"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        cmin, cmax = base.PreprocessorPlugin.components_interval(*args, **kwargs)
        return [params.Integer("n_components", cmin, cmax)]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "PCAPlugin":
        n_components = min(self.n_components, X.shape[0], X.shape[1])

        self.model = PCA(n_components=n_components, random_state=self.random_state)

        self.model.fit(X, *args, **kwargs)

        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(
            {"model": self.model, "n_components": self.n_components}
        )

    @classmethod
    def load(cls, buff: bytes) -> "PCAPlugin":
        args = serialization.load_model(buff)
        return cls(**args)


plugin = PCAPlugin
