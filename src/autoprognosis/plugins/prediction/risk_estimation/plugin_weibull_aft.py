# stdlib
from typing import Any, List

# third party
import pandas as pd

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.risk_estimation.base as base
import autoprognosis.plugins.prediction.risk_estimation.helper_lifelines as helper_lifelines
from autoprognosis.utils.pip import install
import autoprognosis.utils.serialization as serialization

for retry in range(2):
    try:
        # third party
        from lifelines import WeibullAFTFitter

        break
    except ImportError:
        depends = ["lifelines"]
        install(depends)


class WeibullAFTPlugin(base.RiskEstimationPlugin):
    """Weibull AFT plugin for survival analysis.

    Args:
        alpha: float
            the level in the confidence intervals.
        l1_ratio: float
            the penalizer coefficient to the size of the coefficients.
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
        >>> plugin = Predictions(category="risk_estimation").get("weibull_aft")
        >>> plugin.fit(X, T, Y)
        >>>
        >>> eval_time_horizons = [int(T[Y.iloc[:] == 1].quantile(0.50))]
        >>> plugin.predict(X, eval_time_horizons)

    """

    def __init__(
        self,
        alpha: float = 0.05,
        l1_ratio: float = 0,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if model:
            self.model = model
            return

        self.model = helper_lifelines.LifelinesWrapper(
            WeibullAFTFitter(alpha=alpha, l1_ratio=l1_ratio)
        )

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "WeibullAFTPlugin":
        self.model.fit(X, *args)

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    @staticmethod
    def name() -> str:
        return "weibull_aft"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Float("alpha", 0.01, 1.0),
            params.Float("l1_ratio", 0, 0.2),
        ]

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "WeibullAFTPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = WeibullAFTPlugin
