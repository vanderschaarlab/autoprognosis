# stdlib
import copy
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
import shap

# autoprognosis absolute
from autoprognosis.plugins.explainers.base import ExplainerPlugin
from autoprognosis.utils.distributions import enable_reproducible_results


class KernelSHAPPlugin(ExplainerPlugin):
    """
    Interpretability plugin based on KernelSHAP.

    Args:
        estimator: model. The model to explain.
        X: dataframe. Training set
        y: dataframe. Training labels
        task_type: str. classification or risk_estimation
        prefit: bool. If true, the estimator won't be trained.
        n_epoch: int. training epochs
        subsample: int. Number of samples to use.
        time_to_event: dataframe. Used for risk estimation tasks.
        eval_times: list. Used for risk estimation tasks.

    Example:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>>from autoprognosis.plugins.explainers import Explainers
        >>> from autoprognosis.plugins.prediction.classifiers import Classifiers
        >>>
        >>> X, y = load_iris(return_X_y=True)
        >>>
        >>> X = pd.DataFrame(X)
        >>> y = pd.Series(y)
        >>>
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> model = Classifiers().get("logistic_regression")
        >>>
        >>> explainer = Explainers().get(
        >>>     "kernel_shap",
        >>>     model,
        >>>     X_train,
        >>>     y_train,
        >>>     task_type="classification",
        >>> )
        >>>
        >>> explainer.explain(X_test)
    """

    def __init__(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        task_type: str = "classification",
        feature_names: Optional[List] = None,
        subsample: int = 10,
        prefit: bool = False,
        n_epoch: int = 10000,
        # risk estimation
        time_to_event: Optional[pd.DataFrame] = None,  # for survival analysis
        eval_times: Optional[List] = None,  # for survival analysis
        random_state: int = 0,
    ) -> None:
        if task_type not in ["classification", "regression", "risk_estimation"]:
            raise RuntimeError("invalid task type")
        enable_reproducible_results(random_state)

        self.feature_names = (
            feature_names if feature_names is not None else pd.DataFrame(X).columns
        )

        X = pd.DataFrame(X, columns=self.feature_names)
        X_summary = shap.kmeans(X, subsample)
        model = copy.deepcopy(estimator)
        self.task_type = task_type

        if task_type == "classification":
            if not prefit:
                model.fit(X, y)

            def model_fn(X: pd.DataFrame) -> pd.DataFrame:
                X = pd.DataFrame(X, columns=self.feature_names)
                return model.predict_proba(X)

            self.explainer = shap.KernelExplainer(
                model_fn, X_summary, feature_names=self.feature_names
            )
        elif task_type == "regression":
            if not prefit:
                model.fit(X, y)

            def model_fn(X: pd.DataFrame) -> pd.DataFrame:
                X = pd.DataFrame(X, columns=self.feature_names)
                return model.predict(X)

            self.explainer = shap.KernelExplainer(
                model_fn, X_summary, feature_names=self.feature_names
            )
        elif task_type == "risk_estimation":
            if time_to_event is None or eval_times is None:
                raise RuntimeError("Invalid input for risk estimation interpretability")

            if not prefit:
                model.fit(X, time_to_event, y)

            def model_fn(X: pd.DataFrame) -> pd.DataFrame:
                X = pd.DataFrame(X, columns=self.feature_names)
                out = np.asarray(model.predict(X, eval_times))[:, -1]
                return out

            self.explainer = shap.KernelExplainer(
                model_fn, X_summary, feature_names=self.feature_names
            )

    def plot(self, X: pd.DataFrame) -> None:  # type: ignore
        shap_values = self.explainer.shap_values(X)
        shap.summary_plot(shap_values, X)

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X, columns=self.feature_names)
        importance = np.asarray(self.explainer.shap_values(X))
        if self.task_type == "classification":
            importance = importance[1, :]

        return importance

    @staticmethod
    def name() -> str:
        return "kernel_shap"

    @staticmethod
    def pretty_name() -> str:
        return "Kernel SHAP"


plugin = KernelSHAPPlugin
