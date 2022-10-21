# stdlib
import copy
from typing import Any, List, Optional, Union

# third party
import pandas as pd

# autoprognosis absolute
from autoprognosis.plugins.explainers.base import ExplainerPlugin
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        import shap

        break
    except ImportError:
        depends = ["shap"]
        install(depends)


class ShapPermutationSamplerPlugin(ExplainerPlugin):
    """
    Interpretability plugin based on ShapPermutation sampler.

    Args:
        estimator: model. The model to explain.
        X: dataframe. Training set
        y: dataframe. Training labels
        task_type: str. classification of risk_estimation
        prefit: bool. If true, the estimator won't be trained.
        n_epoch: int. training epochs
        subsample: int. Number of samples to use.
        time_to_event: dataframe. Used for risk estimation tasks.
        eval_times: list. Used for risk estimation tasks.
    """

    def __init__(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        feature_names: Optional[List] = None,
        task_type: str = "classification",
        n_epoch: int = 10000,
        # for survival analysis
        time_to_event: Optional[pd.DataFrame] = None,  # for survival analysis
        eval_times: Optional[List] = None,  # for survival analysis
        prefit: bool = False,
        random_state: int = 0,
    ) -> None:

        if task_type not in [
            "classification",
            "risk_estimation",
        ]:
            raise RuntimeError(f"Invalid task type {task_type}")

        self.task_type = task_type
        self.feature_names = (
            feature_names if feature_names is not None else pd.DataFrame(X).columns
        )
        super().__init__(self.feature_names)

        model = copy.deepcopy(estimator)

        if task_type == "classification":
            if not prefit:
                model.fit(X, y)

            def model_fn(X: pd.DataFrame) -> pd.DataFrame:
                return model.predict_proba(X)

        elif task_type == "risk_estimation":
            if time_to_event is None or eval_times is None:
                raise RuntimeError("invalid input for risk estimation interpretability")

            if not prefit:
                model.fit(X, time_to_event, y)

            def model_fn(X: pd.DataFrame) -> pd.DataFrame:
                return model.predict(X, eval_times)

        self.explainer = shap.explainers.Permutation(
            model_fn,
            X,
            feature_names=self.feature_names,
        )

    def explain(self, X: pd.DataFrame, max_evals: Union[int, str] = "auto") -> Any:
        expl = self.explainer(X, max_evals=max_evals, silent=True)
        if self.task_type == "classification":
            out = expl[..., 1]
        else:
            out = expl

        return out.values

    @staticmethod
    def name() -> str:
        return "shap_permutation_sampler"

    @staticmethod
    def pretty_name() -> str:
        return "SHAP Permutation Sampler"


plugin = ShapPermutationSamplerPlugin
