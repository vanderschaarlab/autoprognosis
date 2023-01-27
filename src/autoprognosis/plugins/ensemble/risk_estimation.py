# stdlib
import copy
from typing import Any, Dict, List, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.hooks import DefaultHooks, Hooks
import autoprognosis.logger as log
from autoprognosis.plugins.explainers import Explainers

EPS = 10**-8


class RiskEnsemble:
    """
    Weighted risk ensemble.

    Args:
        models: List [N]
            List of base models.
        weights: List (time_horizons|, N)
            list of weights for each model and each time horizon.
        time_horizons: List
            List of time horizons used for evaluation.
        explainer_plugins: List
            List of explainers attached to the ensemble.
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
        X = pd.DataFrame(X).reset_index(drop=True)
        T = pd.Series(T).reset_index(drop=True)
        Y = pd.Series(Y).reset_index(drop=True)

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

    def is_fitted(self) -> bool:
        _fitted = True
        for model in self.models:
            _fitted = _fitted and model.is_fitted()

        return _fitted

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


class RiskEnsembleCV(RiskEnsemble):
    """
    Cross-validated Weighted ensemble, with uncertainity prediction support

    Args:
        ensemble: List
            Base ensembles. Excludes models/weights
        models: List [N]
            List of base models.
        weights: List (time_horizons, N)
            list of weights for each model and each time horizon.
        time_horizons: List
            List of time horizons used for evaluation.
        explainer_plugins: List
            List of explainers attached to the ensemble.
    """

    def __init__(
        self,
        time_horizons: List,
        ensemble: Optional[RiskEnsemble] = None,
        models: Optional[List] = None,
        weights: Optional[List[float]] = None,
        explainer_plugins: List = [],
        explanations_model: Optional[Dict] = None,
        explanations_nepoch: int = 10000,
        n_folds: int = 3,
        hooks: Hooks = DefaultHooks(),
    ) -> None:
        if ensemble is None and models is None:
            raise ValueError(
                "Invalid input for RiskEnsembleCV. Provide trained ensemble or raw models."
            )
        if n_folds < 2:
            raise ValueError("Invalid value for n_folds. Must be >= 2.")

        self.time_horizons = time_horizons

        self.n_folds = n_folds
        self.seed = 42

        self.explainer_plugins = explainer_plugins
        self.explanations_nepoch = explanations_nepoch
        self.explainers = explanations_model
        self.hooks = hooks

        if ensemble is not None:
            self.models = []
            for fold in range(n_folds):
                self.models.append(copy.deepcopy(ensemble))
        else:
            self.models = []
            if models is None or weights is None:
                raise ValueError(
                    "Invalid input for RiskEnsemble. Provide the models and the weights."
                )
            for folds in range(n_folds):
                self.models.append(
                    RiskEnsemble(models, weights, time_horizons, explainer_plugins=[])
                )

    def fit(self, X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame) -> "RiskEnsemble":
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )
        cv_idx = 0
        for train_index, test_index in skf.split(X, Y):
            self._should_continue()
            X_train = X.iloc[train_index]
            T_train = T.iloc[train_index]
            Y_train = Y.iloc[train_index]

            self.models[cv_idx].fit(X_train, T_train, Y_train)
            cv_idx += 1

        if self.explainers:
            return self

        self.explainers = {}

        for exp in self.explainer_plugins:
            self._should_continue()
            log.info(f"[RiskEnsemble]: train explainer {exp}")
            exp_model = Explainers().get(
                exp,
                copy.deepcopy(self.models[0]),
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

    def is_fitted(self) -> bool:
        _fitted = True
        for model in self.models:
            _fitted = _fitted and model.is_fitted()

        return _fitted

    def predict(
        self,
        X_: pd.DataFrame,
        eval_time_horizons: pd.DataFrame = None,
    ) -> pd.DataFrame:
        results, _ = self.predict_with_uncertainty(X_, eval_time_horizons)

        return results

    def predict_with_uncertainty(
        self,
        X_: pd.DataFrame,
        eval_time_horizons: pd.DataFrame = None,
    ) -> pd.DataFrame:
        results = []

        for model in self.models:
            results.append(np.asarray(model.predict(X_, eval_time_horizons)))

        results = np.asarray(results)
        calibrated_result = np.mean(results, axis=0)
        uncertainity = 1.96 * np.std(results, axis=0) / np.sqrt(len(results))

        return pd.DataFrame(calibrated_result), pd.DataFrame(uncertainity)

    def name(self, short: bool = False) -> str:
        return f"Calibrated  {self.models[0].name()}"
