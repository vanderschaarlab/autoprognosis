# stdlib
from typing import Any, List

# third party
from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.risk_estimation.base as base
from adjutorium.plugins.prediction.risk_estimation.plugin_weibull_aft import (
    WeibullAFTPlugin,
)
import adjutorium.utils.serialization as serialization


class StackedSurvivalCatboost:
    def __init__(self, catboost_params: dict = {}, weibull_params: dict = {}) -> None:
        self._first_layer = CatBoostRegressor(**catboost_params)
        self._second_layer = WeibullAFTPlugin(**weibull_params)

    def fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "StackedSurvivalCatboost":
        if len(args) < 2:
            raise ValueError("Invalid input for fit. Expecting X, T and Y.")
        T = args[0]
        E = args[1]

        (X_train, X_valid, E_train, E_valid, T_train, T_valid) = train_test_split(
            X, E, T, test_size=0.2, random_state=42
        )

        target_train = np.where(E_train, T_train, -T_train)
        target_valid = np.where(E_valid, T_valid, -T_valid)

        train_pool = Pool(X_train, label=target_train)
        test_pool = Pool(X_valid, label=target_valid)

        self._first_layer.fit(train_pool, eval_set=test_pool)
        train_risk = self._first_layer.predict(train_pool)

        min_positive_value = T_train[T_train > 0].min()
        T_train = np.clip(T_train, min_positive_value, None)

        self._second_layer.fit(train_risk, T_train, E_train, ancillary=True)

        return self

    def predict(self, X: pd.DataFrame, eval_times: List) -> pd.DataFrame:
        risk = self._first_layer.predict(X)

        return self._second_layer.predict(risk, eval_times)


class CatboostRiskEstimationPlugin(base.RiskEstimationPlugin):
    grow_policies = ["Depthwise", "SymmetricTree", "Lossguide"]

    def __init__(
        self,
        learning_rate: float = 1e-2,
        depth: int = 6,
        iterations: int = 1000,
        od_type: str = "Iter",
        od_wait: int = 100,
        border_count: int = 128,
        l2_leaf_reg: float = 1e-4,
        random_strength: float = 0,
        grow_policy: int = 0,
        objective: str = "cox",  # "aft", "cox"
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if model:
            self.model = model
            return

        surv_params = {}
        if objective == "aft":
            surv_params = {
                "loss_function": "SurvivalAft:dist=Normal;scale=1.0",
                "eval_metric": "SurvivalAft",
            }
        else:
            surv_params = {
                "loss_function": "Cox",
                "eval_metric": "Cox",
            }
        catboost_params = {
            **surv_params,
            "depth": depth,
            "logging_level": "Silent",
            "allow_writing_files": False,
            "used_ram_limit": "6gb",
            "iterations": iterations,
            "od_type": od_type,
            "od_wait": od_wait,
            "border_count": border_count,
            "random_strength": random_strength,
            "grow_policy": CatboostRiskEstimationPlugin.grow_policies[grow_policy],
        }
        self.model = StackedSurvivalCatboost(catboost_params)

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "CatboostRiskEstimationPlugin":
        self.model.fit(X, *args, **kwargs)

        return self

    def _find_nearest(self, array: np.ndarray, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if len(args) < 1:
            raise ValueError("Invalid input for predict. Expecting X and time horizon.")

        time_horizons = args[0]

        return self.model.predict(X, time_horizons)

    @staticmethod
    def name() -> str:
        return "survival_catboost"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "CatboostRiskEstimationPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = CatboostRiskEstimationPlugin
