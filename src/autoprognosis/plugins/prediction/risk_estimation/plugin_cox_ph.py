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
        from lifelines import CoxPHFitter

        break
    except ImportError:
        depends = ["lifelines"]
        install(depends)


class CoxPHPlugin(base.RiskEstimationPlugin):
    def __init__(
        self,
        alpha: float = 0.05,
        penalizer: float = 0.01,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any
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
