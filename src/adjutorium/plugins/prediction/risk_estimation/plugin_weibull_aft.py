# stdlib
from typing import Any, List

# third party
from lifelines import WeibullAFTFitter
import pandas as pd

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.risk_estimation.base as base
import adjutorium.plugins.prediction.risk_estimation.helper_lifelines as helper_lifelines
import adjutorium.utils.serialization as serialization


class WeibullAFTPlugin(base.RiskEstimationPlugin):
    def __init__(
        self, alpha: float = 0.05, l1_ratio: float = 0, model: Any = None, **kwargs: Any
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
