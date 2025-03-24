# stdlib
from typing import Any, List

# third party
import pandas as pd
from lifelines import CoxPHFitter

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.risk_estimation.base as base
import autoprognosis.plugins.prediction.risk_estimation.helper_lifelines as helper_lifelines
import autoprognosis.utils.serialization as serialization


class CoxPHPlugin(base.RiskEstimationPlugin):
    """CoxPH plugin for survival analysis

    Args:
        alpha: float
            the level in the confidence intervals.
        penalizer: float
            Attach a penalty to the size of the coefficients during regression. This improves stability of the estimates and controls for high correlation between covariates.
        random_state: int
            Random seed

    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> from pycox.datasets import metabric
        >>>
        >>> df = metabric.read_df()
        >>> X = df.drop(["duration", "event"], axis=1)
        >>> Y = df["event"]
        >>> T = df["duration"]
        >>>
        >>> plugin = Predictions(category="risk_estimation").get("cox_ph")
        >>> plugin.fit(X, T, Y)
        >>>
        >>> eval_time_horizons = [int(T[Y.iloc[:] == 1].quantile(0.50))]
        >>> plugin.predict(X, eval_time_horizons)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        penalizer: float = 0.01,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if model:
            self.model = model
            return

        self.model = helper_lifelines.LifelinesWrapper(
            CoxPHFitter(alpha=alpha, penalizer=penalizer)
        )

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CoxPHPlugin":
        X = pd.DataFrame(X)
        self.features = X.columns
        self.model.fit(X, *args, **kwargs)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = pd.DataFrame(X)
        X.columns = self.features
        return self.model.predict(X, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "cox_ph"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("alpha", 0.0, 0.1),
            params.Float("penalizer", 0, 0.2),
        ]

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "CoxPHPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = CoxPHPlugin
