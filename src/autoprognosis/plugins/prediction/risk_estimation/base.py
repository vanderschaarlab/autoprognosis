# stdlib
import copy
from typing import Any, List, Optional

# third party
import pandas as pd

# autoprognosis absolute
import autoprognosis.logger as log
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.base as prediction_base


class RiskEstimationPlugin(prediction_base.PredictionPlugin):
    """Base class for the survival analysis plugins.

    It provides the implementation for plugin.Plugin's subtype, _fit and _predict methods.

    Each derived class must implement the following methods(inherited from plugin.Plugin):
        name() - a static method that returns the name of the plugin.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `Params` derived objects.

    If any method implementation is missing, the class constructor will fail.
    """

    def __init__(
        self,
        with_explanations: bool = False,
        explanations_model: Optional[Any] = None,
        explanations_nepoch: int = 10000,
        explanations_nfolds: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.args = kwargs

        self.with_explanations = with_explanations or (explanations_model is not None)
        self.explanations_nepoch = explanations_nepoch
        self.explanations_nfolds = explanations_nfolds
        self.explainer = explanations_model

    @staticmethod
    def subtype() -> str:
        return "risk_estimation"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "RiskEstimationPlugin":
        if len(args) < 2:
            raise ValueError("Invalid input for fit. Expecting X, T and Y.")

        T = args[0]
        Y = args[1]

        X = self._preprocess_training_data(X)

        log.debug(f"Training using {self.fqdn()}, input shape = {X.shape}")
        self._fit(X, *args, **kwargs)
        self._fitted = True

        if self.with_explanations and self.explainer is None:
            log.debug(
                "Training explainer forusing {self.fqdn()}, input shape = {X.shape}"
            )
            if "eval_times" not in kwargs:
                raise RuntimeError("fit requires eval_times")

            # autoprognosis absolute
            from autoprognosis.plugins.explainers.plugin_invase import (
                plugin as explainer,
            )

            self.explainer = explainer(
                copy.deepcopy(self),
                X,
                Y,
                time_to_event=T,
                eval_times=kwargs["eval_times"],
                n_epoch=self.explanations_nepoch,
                n_folds=self.explanations_nfolds,
                prefit=True,
                task_type="risk_estimation",
            )

        log.debug(f"Done training using {self.fqdn()}, input shape = {X.shape}")
        return self

    def explain(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if not self.is_fitted():
            raise RuntimeError("Fit the model first")

        X = self._preprocess_inference_data(X)
        if self.explainer is None:
            raise ValueError("Interpretability is not enabled for this model")

        return self.explainer.explain(X)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        raise NotImplementedError(f"Model {self.name()} doesn't support predict proba")

    def get_args(self) -> dict:
        return self.args
