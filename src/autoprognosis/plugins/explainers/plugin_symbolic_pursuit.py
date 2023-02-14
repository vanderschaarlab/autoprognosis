# stdlib
import copy
from typing import Any, Callable, List, Optional

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.plugins.explainers.base import ExplainerPlugin
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        from symbolic_pursuit.models import SymbolicRegressor

        break
    except ImportError:
        depends = ["symbolic_pursuit"]
        install(depends)


class SymbolicPursuitPlugin(ExplainerPlugin):
    """
    Interpretability plugin based on Symbolic Pursuit.

    Based on the NeurIPS 2020 paper "Learning outside the black-box: at the pursuit of interpretable models".

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
        loss_tol: float. The tolerance for the loss under which the pursuit stops
        ratio_tol: float. A new term is added only if new_loss / old_loss < ratio_tol
        maxiter: float.  Maximum number of iterations for optimization
        eps: float. Number used for numerical stability
        random_state: float. Random seed for reproducibility

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
        >>>     "symbolic_pursuit",
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
        # symbolic pursuit params
        loss_tol: float = 1.0e-3,
        ratio_tol: float = 0.9,
        maxiter: int = 100,
        eps: float = 1.0e-5,
        patience: int = 10,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        if task_type not in ["classification", "risk_estimation", "regression"]:
            raise RuntimeError("invalid task type")

        self.feature_names = (
            feature_names if feature_names is not None else pd.DataFrame(X).columns
        )

        X = pd.DataFrame(X, columns=self.feature_names)
        model = copy.deepcopy(estimator)

        self.task_type = task_type
        self.loss_tol = loss_tol
        self.ratio_tol = ratio_tol
        self.maxiter = maxiter
        self.eps = eps
        self.random_state = random_state

        std_args = {
            "loss_tol": loss_tol,
            "ratio_tol": ratio_tol,
            "random_seed": random_state,
            "maxiter": maxiter,
            "patience": patience,
        }

        if task_type == "classification":
            if not prefit:
                model.fit(X, y)

            self.explainer = SymbolicRegressor(
                **std_args,
                task_type="classification",
            )
            self.explainer.fit(model.predict, X)
        elif task_type == "regression":
            if not prefit:
                model.fit(X, y)

            self.explainer = SymbolicRegressor(
                **std_args,
                task_type="regression",
            )
            self.explainer.fit(model.predict, X)
        elif task_type == "risk_estimation":
            if time_to_event is None or eval_times is None:
                raise RuntimeError("Invalid input for risk estimation interpretability")

            if not prefit:
                model.fit(X, time_to_event, y)

            def model_fn_factory(horizon: int) -> Callable:
                def model_fn(X: pd.DataFrame) -> pd.DataFrame:
                    out = np.asarray(model.predict(X, [horizon])).squeeze()

                    return out

                return model_fn

            self.explainer = SymbolicRegressor(
                **std_args,
                task_type="classification",
            )
            self.explainer.fit(model_fn_factory(eval_times[-1]), X)

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X, columns=self.feature_names)

        results = []
        for idx, row in X.iterrows():
            results.append(self.explainer.get_feature_importance(row.values))

        return np.asarray(results)

    def plot(self, X: pd.DataFrame) -> tuple:  # type: ignore
        return str(self.explainer), self.explainer.get_projections()

    @staticmethod
    def name() -> str:
        return "symbolic_pursuit"

    @staticmethod
    def pretty_name() -> str:
        return "Symbolic Pursuit"


plugin = SymbolicPursuitPlugin
