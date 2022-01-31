# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.risk_estimation.base as base
from adjutorium.utils.pip import install
import adjutorium.utils.serialization as serialization

for retry in range(2):
    try:
        # third party
        from xgbse import XGBSEDebiasedBCE, XGBSEStackedWeibull
        from xgbse.converters import convert_to_structured

        break
    except ImportError:
        depends = ["xgbse"]
        install(depends)


class XGBoostRiskEstimationPlugin(base.RiskEstimationPlugin):
    booster = ["gbtree", "gblinear", "dart"]

    def __init__(
        self,
        n_estimators: int = 100,
        colsample_bynode: float = 0.5,
        max_depth: int = 8,
        subsample: float = 0.5,
        learning_rate: float = 5e-2,
        min_child_weight: int = 50,
        tree_method: str = "hist",
        booster: int = 2,
        random_state: int = 0,
        objective: str = "aft",  # "aft", "cox"
        strategy: str = "weibull",  # "weibull", "debiased_bce"
        model: Any = None,
        hyperparam_search_iterations: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if model:
            self.model = model
            return

        if hyperparam_search_iterations:
            n_estimators = 10 * int(hyperparam_search_iterations)

        surv_params = {}
        if objective == "aft":
            surv_params = {
                "objective": "survival:aft",
                "eval_metric": "aft-nloglik",
                "aft_loss_distribution": "normal",
                "aft_loss_distribution_scale": 1.0,
            }
        else:
            surv_params = {
                "objective": "survival:cox",
                "eval_metric": "cox-nloglik",
            }
        xgboost_params = {
            # survival
            **surv_params,
            **kwargs,
            # basic xgboost
            "n_estimators": n_estimators,
            "colsample_bynode": colsample_bynode,
            "max_depth": max_depth,
            "subsample": subsample,
            "learning_rate": learning_rate,
            "min_child_weight": min_child_weight,
            "verbosity": 0,
            "tree_method": tree_method,
            "booster": XGBoostRiskEstimationPlugin.booster[booster],
            "random_state": random_state,
            "n_jobs": 2,
        }
        lr_params = {
            "C": 1e-3,
            "max_iter": 10000,
        }
        if strategy == "debiased_bce":
            base_model = XGBSEDebiasedBCE(xgboost_params, lr_params)
        elif strategy == "weibull":
            base_model = XGBSEStackedWeibull(xgboost_params)
        else:
            raise ValueError(f"unknown strategy {strategy}")

        self.model = base_model
        # self.model = XGBSEBootstrapEstimator(base_model, n_estimators=20)

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "XGBoostRiskEstimationPlugin":
        if len(args) < 2:
            raise ValueError("Invalid input for fit. Expecting X, T and Y.")

        eval_times = None
        if "eval_times" in kwargs:
            eval_times = kwargs["eval_times"]

        T = args[0]
        E = args[1]

        y = convert_to_structured(T, E)

        (X_train, X_valid, y_train, y_valid) = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(
            X_train,
            y_train,
            num_boost_round=1500,
            validation_data=(X_valid, y_valid),
            early_stopping_rounds=10,
            time_bins=eval_times,
        )

        return self

    def _find_nearest(self, array: np.ndarray, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if len(args) < 1:
            raise ValueError("Invalid input for predict. Expecting X and time horizon.")

        time_horizons = args[0]

        # surv, upper_ci, lower_ci = self.model.predict(X, return_ci = True)
        surv = self.model.predict(X)
        surv = surv.loc[:, ~surv.columns.duplicated()]

        preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])

        time_bins = surv.columns
        for t, eval_time in enumerate(time_horizons):
            nearest = self._find_nearest(time_bins, eval_time)
            preds_[:, t] = np.asarray(1 - surv[nearest])

        return preds_

    @staticmethod
    def name() -> str:
        return "survival_xgboost"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("max_depth", 2, 6),
            params.Integer("min_child_weight", 0, 50),
            params.Categorical("objective", ["aft", "cox"]),
            params.Categorical("strategy", ["weibull", "debiased_bce"]),
        ]

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "XGBoostRiskEstimationPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = XGBoostRiskEstimationPlugin
