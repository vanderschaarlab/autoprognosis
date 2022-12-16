# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd
from sklearn.decomposition import FastICA

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class FastICAPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for dimensionality reduction based on Independent Component Analysis algorithm.

    Method:
        Independent component analysis separates a multivariate signal into additive subcomponents that are maximally independent.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

    Args:
        n_components: int
            Number of components to use.
    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors(category="dimensionality_reduction").get("fast_ica")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(
        self,
        model: Any = None,
        random_state: int = 0,
        n_components: int = 2,
        max_iter=10000,
    ) -> None:
        super().__init__()
        self.random_state = random_state
        self.n_components = n_components
        self.model: Optional[FastICA] = None

        if model:
            self.model = model

    @staticmethod
    def name() -> str:
        return "fast_ica"

    @staticmethod
    def subtype() -> str:
        return "dimensionality_reduction"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        cmin, cmax = base.PreprocessorPlugin.components_interval(*args, **kwargs)
        return [params.Integer("n_components", cmin, cmax)]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "FastICAPlugin":
        n_components = min(self.n_components, X.shape[0], X.shape[1])
        self.model = FastICA(
            n_components=n_components,
            random_state=self.random_state,
            max_iter=10000,
            tol=1e-2,
            whiten="unit-variance",
        )

        self.model.fit(X, *args, **kwargs)
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(
            {"model": self.model, "n_components": self.n_components}
        )

    @classmethod
    def load(cls, buff: bytes) -> "FastICAPlugin":
        model = serialization.load_model(buff)
        return cls(**model)


plugin = FastICAPlugin
