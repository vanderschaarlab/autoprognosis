# stdlib
import copy
from typing import Any, List, Optional, Union

# third party
import pandas as pd
import shap

# adjutorium absolute
from adjutorium.plugins.explainers.base import ExplainerPlugin


class ShapPermutationSamplerPlugin(ExplainerPlugin):
    def __init__(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        feature_names: Optional[List] = None,
        task_type: str = "classification",
        n_epoch: int = 10000,
        # for treatment effects
        w: Optional[pd.DataFrame] = None,
        y_full: Optional[pd.DataFrame] = None,  # for treatment effects
        # for survival analysis
        time_to_event: Optional[pd.DataFrame] = None,  # for survival analysis
        eval_times: Optional[List] = None,  # for survival analysis
        prefit: bool = False,
    ) -> None:

        assert task_type in [
            "classification",
            "treatments",
            "risk_estimation",
        ], f"Invalid task type {task_type}"

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

        elif task_type == "treatments":
            assert w is not None
            assert y_full is not None

            if not prefit:
                model.fit(X, time_to_event, y)

            def model_fn(X: pd.DataFrame) -> pd.DataFrame:
                return model.predict(X)

        elif task_type == "risk_estimation":
            assert time_to_event is not None
            assert eval_times is not None

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
