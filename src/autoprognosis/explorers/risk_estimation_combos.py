# stdlib
import copy
import time
from typing import List

# third party
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_risk_estimation_names,
)
from autoprognosis.explorers.core.optimizer import EarlyStoppingExceeded, create_study
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
    ) -> None:
        self.time_horizons = time_horizons
        self.num_ensemble_iter = num_ensemble_iter
        self.num_iter = num_iter
        self.timeout = timeout
        self.ensemble_size = ensemble_size
        self.CV = CV
        self.hooks = hooks

        self.study_name = study_name

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
    ) -> List:
        self._should_continue()

        skf = StratifiedKFold(n_splits=self.CV, shuffle=True, random_state=seed)

        ensemble_folds = []

        for train_index, _ in skf.split(X, Y):

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
    ) -> List[float]:
        self._should_continue()

        study, pruner = create_study(
            load_if_exists=False,
            storage_type="none",
            study_name=f"{self.study_name}_risk_estimation_exploration_ensemble_{time_horizon}",
        )

        pretrained_models = self.pretrain_for_cv(ensemble, X, T, Y, time_horizon)

        def objective(trial: optuna.Trial) -> float:
            self._should_continue()
            start = time.time()

            weights = [
                trial.suggest_int(f"weight_{idx}", 0, 10)
                for idx in range(len(ensemble))
            ]
            pruner.check_trial(trial)

            weights = weights / (np.sum(weights) + EPS)

            cv_folds = []
            for fold in pretrained_models:
                cv_folds.append(RiskEnsemble(fold, [weights], [time_horizon]))

            metrics = evaluate_survival_estimator(
                cv_folds, X, T, Y, [time_horizon], pretrained=True
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
                f"Trial {trial.number}: ensemble {cv_folds[0].name()} : results {metrics['clf']['c_index'][0]}"
            )
            score = metrics["clf"]["c_index"][0] - metrics["clf"]["brier_score"][0]

            pruner.report_score(score)

            return score

        if not skip_recap:
            initial_trials = []

            trial_template = {}
            for idx in range(len(ensemble)):
                trial_template[f"weight_{idx}"] = 0

            for idx in range(len(ensemble)):
                local_trial = copy.deepcopy(trial_template)
                local_trial[f"weight_{idx}"] = 1
                initial_trials.append(local_trial)

            for trial in initial_trials:
                study.enqueue_trial(trial)

        try:
            study.optimize(
                objective, n_trials=self.num_ensemble_iter, timeout=self.timeout
            )
        except EarlyStoppingExceeded:
            log.info("early stopping triggered for ensemble search")

        weights = []
        for idx in range(len(ensemble)):
            weights.append(study.best_trial.params[f"weight_{idx}"])
        weights = weights / (np.sum(weights) + EPS)
        log.info(
            f"Best trial for ensemble {time_horizon}: {study.best_value} for {weights}"
        )
        return weights

    def search(
        self,
        X: pd.DataFrame,
        T: pd.DataFrame,
        Y: pd.DataFrame,
        skip_recap: bool = False,
    ) -> RiskEnsemble:
        self._should_continue()

        best_horizon_models = self.estimator_seeker.search(X, T, Y)
        all_models = [
            model for horizon_models in best_horizon_models for model in horizon_models
        ]

        weights: List[List[float]] = []

        for idx, horizon in enumerate(self.time_horizons):
            self._should_continue()

            local_weights = self.search_weights(
                all_models, X, T, Y, horizon, skip_recap
            )
            weights.append(local_weights)

        result = RiskEnsemble(all_models, weights, self.time_horizons)
        log.info(f"Final ensemble: {result.name()}")

        return result
