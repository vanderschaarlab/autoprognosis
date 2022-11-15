# stdlib
import copy
import time
from typing import List, Optional

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_risk_estimation_names,
)
from autoprognosis.explorers.core.optimizer import EnsembleOptimizer
from autoprognosis.explorers.hooks import DefaultHooks
from autoprognosis.hooks import Hooks
import autoprognosis.logger as log
from autoprognosis.plugins.ensemble.risk_estimation import RiskEnsemble
from autoprognosis.utils.tester import evaluate_survival_estimator

# autoprognosis relative
from .risk_estimation import RiskEstimatorSeeker

EPS = 10 ** -8


class RiskEnsembleSeeker:
    """
    AutoML core logic for risk estimation ensemble search.

    Args:
        study_name: str.
            Study ID, used for caching.
        time_horizons: list.
            list of time horizons.
        num_iter: int.
            Number of optimization trials.
        num_ensemble_iter: int.
            Number of optimization trials for the ensemble weights.
        timeout: int.
            Max wait time(in seconds) for the optimization output.
        CV: int.
            Number of folds to use for evaluation
        feature_scaling: list.
            Plugins to use in the pipeline for preprocessing.
        estimators: list.
            Plugins to use in the pipeline for risk prediction.
        imputers: list.
            Plugins to use in the pipeline for imputation.
        hooks: Hooks.
            Custom callbacks to be notified about the search progress.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        time_horizons: List[int],
        num_iter: int = 50,
        num_ensemble_iter: int = 100,
        timeout: int = 360,
        CV: int = 3,
        estimators: List[str] = default_risk_estimation_names,
        ensemble_size: int = 2,
        imputers: List[str] = [],
        feature_scaling: List[str] = default_feature_scaling_names,
        hooks: Hooks = DefaultHooks(),
        optimizer_type: str = "bayesian",
    ) -> None:
        self.time_horizons = time_horizons
        self.num_ensemble_iter = num_ensemble_iter
        self.num_iter = num_iter
        self.timeout = timeout
        self.ensemble_size = ensemble_size
        self.CV = CV
        self.hooks = hooks

        self.study_name = study_name
        self.optimizer_type = optimizer_type

        self.estimator_seeker = RiskEstimatorSeeker(
            study_name,
            time_horizons=time_horizons,
            num_iter=num_iter,
            CV=CV,
            top_k=ensemble_size,
            timeout=timeout,
            estimators=estimators,
            hooks=hooks,
            feature_scaling=feature_scaling,
            imputers=imputers,
            optimizer_type=optimizer_type,
        )

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Risk estimation combos search cancelled")

    def pretrain_for_cv(
        self,
        ensemble: List,
        X: pd.DataFrame,
        T: pd.DataFrame,
        Y: pd.DataFrame,
        time_horizon: int,
        seed: int = 0,
        group_ids: Optional[str] = None,
    ) -> List:
        self._should_continue()

        if group_ids is not None:
            skf = StratifiedGroupKFold(
                n_splits=self.CV, shuffle=True, random_state=seed
            )
        else:
            skf = StratifiedKFold(n_splits=self.CV, shuffle=True, random_state=seed)

        ensemble_folds = []

        for train_index, _ in skf.split(X, Y, groups=group_ids):

            X_train = X.loc[X.index[train_index]]
            Y_train = Y.loc[Y.index[train_index]]
            T_train = T.loc[T.index[train_index]]

            local_fold = []
            for estimator in ensemble:
                model = copy.deepcopy(estimator)
                model.fit(X_train, T_train, Y_train)
                local_fold.append(model)
            ensemble_folds.append(local_fold)
        return ensemble_folds

    def search_weights(
        self,
        ensemble: List,
        X: pd.DataFrame,
        T: pd.DataFrame,
        Y: pd.DataFrame,
        time_horizon: int,
        skip_recap: bool = False,
        group_ids: Optional[pd.Series] = None,
    ) -> List[float]:
        self._should_continue()

        pretrained_models = self.pretrain_for_cv(
            ensemble, X, T, Y, time_horizon, group_ids=group_ids
        )

        def evaluate(weights: list) -> float:
            self._should_continue()
            start = time.time()

            cv_folds = []
            for fold in pretrained_models:
                cv_folds.append(RiskEnsemble(fold, [weights], [time_horizon]))

            metrics = evaluate_survival_estimator(
                cv_folds, X, T, Y, [time_horizon], pretrained=True, group_ids=group_ids
            )

            self.hooks.heartbeat(
                topic="risk_estimation",
                subtopic="ensemble_search",
                event_type="performance",
                name=cv_folds[0].name(short=True),
                duration=time.time() - start,
                horizon=time_horizon,
                aucroc=metrics["str"]["aucroc"],
                cindex=metrics["str"]["c_index"],
                brier_score=metrics["str"]["brier_score"],
            )

            log.debug(
                f"Ensemble {cv_folds[0].name()} : results {metrics['clf']['c_index'][0]}"
            )
            return metrics["clf"]["c_index"][0] - metrics["clf"]["brier_score"][0]

        study = EnsembleOptimizer(
            study_name=f"{self.study_name}_risk_estimation_exploration_ensemble_{time_horizon}",
            ensemble_len=len(ensemble),
            evaluation_cbk=evaluate,
            optimizer_type=self.optimizer_type,
            n_trials=self.num_iter,
            timeout=self.timeout,
            skip_recap=skip_recap,
        )

        best_score, selected_weights = study.evaluate()

        weights = []
        for idx in range(len(ensemble)):
            weights.append(selected_weights[f"weight_{idx}"])
        weights = weights / (np.sum(weights) + EPS)
        log.info(f"Best trial for ensemble: {best_score} for {weights}")

        return weights

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def search(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        Y: pd.Series,
        skip_recap: bool = False,
        group_ids: Optional[pd.Series] = None,
    ) -> RiskEnsemble:
        self._should_continue()

        best_horizon_models = self.estimator_seeker.search(X, T, Y, group_ids=group_ids)
        all_models = [
            model for horizon_models in best_horizon_models for model in horizon_models
        ]

        weights: List[List[float]] = []

        for idx, horizon in enumerate(self.time_horizons):
            self._should_continue()

            local_weights = self.search_weights(
                all_models,
                X,
                T,
                Y,
                horizon,
                skip_recap=skip_recap,
                group_ids=group_ids,
            )
            weights.append(local_weights)

        result = RiskEnsemble(all_models, weights, self.time_horizons)
        log.info(f"Final ensemble: {result.name()}")

        return result
