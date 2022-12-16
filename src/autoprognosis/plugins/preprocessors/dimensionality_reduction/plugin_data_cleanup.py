# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# autoprognosis absolute
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.preprocessors.base as base
import autoprognosis.utils.serialization as serialization


class DataCompressionPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin used for droping constant features, and for fixing multicollinearity issues.

    Args:

        threshold: float
            The variance threshold.
        vif_threshold: float
            The limit for the VIF values.
        drop_variance:
            enable VarianceThreshold
        drop_multicollinearity:
            enable multicollinearity filtering.

    Example:
        >>> from autoprognosis.plugins.preprocessors import Preprocessors
        >>> plugin = Preprocessors(category="dimensionality_reduction").get("data_cleanup")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_transform(X, y)
    """

    def __init__(
        self,
        threshold: float = 0,
        vif_threshold: float = 10,
        drop_variance: bool = True,
        drop_multicollinearity: bool = True,
    ) -> None:
        super().__init__()

        self.var_threshold = VarianceThreshold(threshold=threshold)
        self.vif_threshold = vif_threshold
        self.scaler = MinMaxScaler()
        self.drop_variance = drop_variance
        self.drop_multicollinearity = drop_multicollinearity

    @staticmethod
    def name() -> str:
        return "data_cleanup"

    @staticmethod
    def subtype() -> str:
        return "dimensionality_reduction"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _compute_vif(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy().fillna(0)

        vif = pd.DataFrame()
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns

        log.debug(f"[Data cleanup] VIF = {vif}")
        return vif

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "DataCompressionPlugin":
        X = X.copy()
        self.columns = X.columns

        X = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)

        self.drop: List[str] = []
        orig_index = X.index

        if self.drop_variance:
            X = self.var_threshold.fit_transform(X)
            X = pd.DataFrame(
                X, columns=self.var_threshold.get_feature_names_out(), index=orig_index
            )

        if self.drop_multicollinearity and X.shape[1] > 1:
            vif = self._compute_vif(X)
            vif_max = vif.loc[vif["VIF"].idxmax()]
            drop = []
            while vif_max["VIF"] > self.vif_threshold:
                drop.append(vif_max["features"])
                eval_X = X.drop(columns=drop)

                if eval_X.shape[1] <= 1:
                    break

                vif = self._compute_vif(eval_X)
                vif_max = vif.loc[vif["VIF"].idxmax()]

            self.drop = drop

        return self

    def _transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = X.copy()
        X = pd.DataFrame(self.scaler.transform(X), columns=self.columns, index=X.index)

        if self.drop_variance:
            orig_index = X.index
            X = self.var_threshold.transform(X)
            X = pd.DataFrame(
                X, columns=self.var_threshold.get_feature_names_out(), index=orig_index
            )

        if self.drop_multicollinearity:
            X = X.drop(columns=self.drop)

        return X

    def save(self) -> bytes:
        return serialization.save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "DataCompressionPlugin":
        return serialization.load_model(buff)


plugin = DataCompressionPlugin
