# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.cluster import FeatureAgglomeration

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class FeatureAgglomerationPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin for dimensionality reduction based on Feature Agglomeration algorithm.

    Method:
        FeatureAgglomeration uses agglomerative clustering to group together features that look very similar, thus decreasing the number of features.

    Reference:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html

    Args:
        n_clusters: int
            Number of clusters to find.

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors(category="dimensionality_reduction").get("feature_agglomeration")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(
        self, model: Any = None, random_state: int = 0, n_clusters: int = 2
    ) -> None:
        super().__init__()
        if model:
            self.model = model
            return
        self.model = FeatureAgglomeration(n_clusters=n_clusters)

    @staticmethod
    def name() -> str:
        return "feature_agglomeration"

    @staticmethod
    def subtype() -> str:
        return "dimensionality_reduction"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        cmin, cmax = base.PreprocessorPlugin.components_interval(*args, **kwargs)
        return [params.Integer("n_clusters", cmin, cmax)]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "FeatureAgglomerationPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.model.transform(X)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "FeatureAgglomerationPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = FeatureAgglomerationPlugin
