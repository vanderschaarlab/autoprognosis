# stdlib
import copy
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
import shap

# adjutorium absolute
from adjutorium.plugins.explainers.base import ExplainerPlugin


class KernelSHAPPlugin(ExplainerPlugin):
    def __init__(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        task_type: str = "classification",
        feature_names: Optional[List] = None,
        subsample: int = 1000,
        prefit: bool = False,
        n_epoch: int = 10000,
        # Treatment effects
        w: Optional[pd.DataFrame] = None,
        y_full: Optional[pd.DataFrame] = None,  # for treatment effects
        # risk estimation
        time_to_event: Optional[pd.DataFrame] = None,  # for survival analysis
        eval_times: Optional[List] = None,  # for survival analysis
    ) -> None:
        assert task_type in ["classification", "treatments", "risk_estimation"]
        self.feature_names = (
            feature_names if feature_names is not None else pd.DataFrame(X).columns
        )

        X = pd.DataFrame(X, columns=self.feature_names)
        X_summary = shap.sample(X, subsample)
        model = copy.deepcopy(estimator)
        self.task_type = task_type

        if task_type == "classification":
            if not prefit:
                model.fit(X, y)
            self.explainer = shap.KernelExplainer(
                model.predict_proba, X_summary, feature_names=self.feature_names
            )
        elif task_type == "treatments":
            assert w is not None
            assert y_full is not None

            if not prefit:
                model.fit(X, w, y)

            self.explainer = shap.KernelExplainer(
                model.predict, X_summary, feature_names=self.feature_names
            )
        elif task_type == "risk_estimation":
            assert time_to_event is not None
            assert eval_times is not None

            if not prefit:
                model.fit(X, time_to_event, y)

            def model_fn(X: pd.DataFrame) -> pd.DataFrame:
                out = np.asarray(model.predict(X, eval_times))

                return out

            self.explainer = shap.KernelExplainer(
                model_fn, X_summary, feature_names=self.feature_names
            )

    def plot(self, shap_values: Any, X: pd.DataFrame) -> None:  # type: ignore
        X = pd.DataFrame(X, columns=self.feature_names)
        shap.summary_plot(shap_values, X)

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X, columns=self.feature_names)
        results = []
        for index, row in X.iterrows():
            importance = np.asarray(self.explainer.shap_values(row)).T
            if self.task_type == "classification":
                importance = importance[:, 1]
            results.append(importance)

        return np.asarray(results)

    @staticmethod
    def name() -> str:
        return "kernel_shap"

    @staticmethod
    def pretty_name() -> str:
        return "Kernel SHAP"


plugin = KernelSHAPPlugin
