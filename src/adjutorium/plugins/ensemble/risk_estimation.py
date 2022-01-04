# stdlib
import copy
from typing import Any, Dict, List, Optional

# third party
import numpy as np
import pandas as pd

# adjutorium absolute
from adjutorium.exceptions import StudyCancelled
from adjutorium.explorers.hooks import DefaultHooks
from adjutorium.hooks import Hooks
import adjutorium.logger as log
from adjutorium.plugins.explainers import Explainers

EPS = 10 ** -8


class RiskEnsemble:
    """
    Weighted risk ensemble.

    Args:
        models: list of base models.
        weights: list of weights for each model and each time horizon.
        time_horizons: list of time horizons used for evaluation.
        explainer_plugins: list of explainers attached to the ensemble.
    """

    def __init__(
        self,
        models: List,
        weights: np.ndarray,
        time_horizons: List,
        explainer_plugins: List = [],
        explanations_model: Optional[Dict] = None,
        explanations_nepoch: int = 10000,
        hooks: Hooks = DefaultHooks(),
    ) -> None:
        if len(weights) != len(time_horizons):
            raise RuntimeError("RiskEnsemble: weights, time_horizon shape mismatch")
        if len(models) != len(weights[0]):
            raise RuntimeError("RiskEnsemble: models, weights shape mismatch")

        try:
            self.models = copy.deepcopy(models)
        except BaseException:
            self.models = models

        self.time_horizons = time_horizons
        self.weights = np.asarray(weights)
        self.weights = self.weights / np.sum(self.weights + EPS, axis=-1).reshape(-1, 1)

        self.explainer_plugins = explainer_plugins
        self.explanations_nepoch = explanations_nepoch
        self.explainers = explanations_model
        self.hooks = hooks

        try:
            self._compress_models()
        except BaseException:
            pass

    def _compress_models(self) -> None:
        compressed: dict = {}
        for idx, mod in enumerate(self.models):
            if str(mod.args) not in compressed:
                compressed[str(mod.args)] = []
            compressed[str(mod.args)].append(idx)

        compressed_weights: list = []
        for hidx, _ in enumerate(self.weights):
            compressed_weights.append([])

        compressed_models = []

        for group in compressed:
            indices = compressed[group]

            raw_model = copy.deepcopy(self.models[compressed[group][0]])

            compressed_models.append(raw_model)
            for hidx, horiz_weights in enumerate(self.weights):
                compressed_w = 0

                for idx in indices:
                    compressed_w += horiz_weights[idx]

                compressed_weights[hidx].append(compressed_w)

        self.models = compressed_models
        self.weights = np.asarray(compressed_weights)

    def enable_explainer(
        self,
        explainer_plugins: list = [],
        explanations_nepoch: int = 10000,
    ) -> None:
        self.explainer_plugins = explainer_plugins
        self.explanations_nepoch = explanations_nepoch

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("risk estimation ensemble: cancelled")

    def fit(self, X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame) -> "RiskEnsemble":
        for model in self.models:
            self._should_continue()
            log.info(f"[RiskEnsemble]: train {model.name()} {model.get_args()}")
            model.fit(X, T, Y)

        if self.explainers:
            return self

        self.explainers = {}

        for exp in self.explainer_plugins:
            self._should_continue()
            log.info(f"[RiskEnsemble]: train explainer {exp}")
            exp_model = Explainers().get(
                exp,
                copy.deepcopy(self),
                X,
                Y,
                time_to_event=T,
                eval_times=self.time_horizons,
                n_epoch=self.explanations_nepoch,
                prefit=True,
                task_type="risk_estimation",
            )
            self.explainers[exp] = exp_model

        return self

    def predict(
        self,
        X_: pd.DataFrame,
        eval_time_horizons: pd.DataFrame = None,
    ) -> pd.DataFrame:
        if eval_time_horizons is None:
            eval_time_horizons = self.time_horizons

        pred = np.zeros([np.shape(X_)[0], len(eval_time_horizons)])

        nearest_fit = []
        for eval_time in eval_time_horizons:
            nearest_fit.append(
                (np.abs(np.asarray(self.time_horizons) - eval_time)).argmin()
            )

        local_predicts = []
        for midx, model in enumerate(self.models):
            log.debug(f"[RiskEnsemble] predict for {model.name} on {X_.shape}")
            local_predicts.append(model.predict(X_, eval_time_horizons))

        for midx, local_pred in enumerate(local_predicts):
            for tidx, actual_tidx in enumerate(nearest_fit):
                tmp_pred = np.copy(local_pred)

                for tt in range(len(eval_time_horizons), 1, -1):
                    tt = tt - 1
                    tmp_pred[:, tt] -= tmp_pred[:, tt - 1]

                pred[:, tidx:] += self.weights[actual_tidx, midx] * tmp_pred[:, [tidx]]
                # add the increment by the model selected at the current timehorizon

        return pd.DataFrame(pred)

    def explain(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if self.explainers is None:
            raise ValueError("Interpretability is not enabled for this model")

        results = {}
        for exp in self.explainers:
            results[exp] = self.explainers[exp].explain(X)

        return results

    def name(self, short: bool = False) -> str:
        ens_name = []
        for horizon in self.weights:
            local_name = []
            for idx in range(len(self.models)):
                if horizon[idx] == 0:
                    continue
                name = f"{round(horizon[idx], 2)} * {self.models[idx].name()}"
                if hasattr(self.models[idx], "get_args") and not short:
                    name += f"({self.models[idx].get_args()})"
                local_name.append(name)

            horiz_name = " + ".join(local_name)
            ens_name.append(horiz_name)
        return str(ens_name)
