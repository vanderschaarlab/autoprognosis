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
    default_feature_selection_names,
    default_risk_estimation_names,
)
from autoprognosis.explorers.core.optimizer import EnsembleOptimizer
from autoprognosis.hooks import DefaultHooks, Hooks
import autoprognosis.logger as log
from autoprognosis.plugins.ensemble.risk_estimation import RiskEnsemble
from autoprognosis.utils.tester import evaluate_survival_estimator

# autoprognosis relative
from .risk_estimation import RiskEstimatorSeeker

EPS = 10**-8


class RiskEnsembleSeeker:
    """
    AutoML core logic for risk estimation ensemble search.

    Args:
        study_name: str.
            Study ID, used for caching.
        time_horizons: list.
            list of time horizons.
        num_iter: int.
            Maximum Number of optimization trials. This is the limit of trials for each base estimator in the "risk_estimators" list, used in combination with the "timeout" parameter. For each estimator, the search will end after "num_iter" trials or "timeout" seconds.
        num_ensemble_iter: int.
            Number of optimization trials for the ensemble weights.
        timeout: int.
            Maximum wait time(seconds) for each estimator hyperparameter search. This timeout will apply to each estimator in the "risk_estimators" list.
        n_folds_cv: int.
            Number of folds to use for evaluation
        feature_scaling: list.
            Plugin search pool to use in the pipeline for scaling. Defaults to : ['maxabs_scaler', 'scaler', 'feature_normalizer', 'normal_transform', 'uniform_transform', 'nop', 'minmax_scaler']
            Available plugins, retrieved using `Preprocessors(category="feature_scaling").list_available()`:
                - 'maxabs_scaler'
                - 'scaler'
                - 'feature_normalizer'
                - 'normal_transform'
                - 'uniform_transform'
                - 'nop' # empty operation
                - 'minmax_scaler'
        feature_selection: list.
            Plugin search pool to use in the pipeline for feature selection. Defaults ["nop", "variance_threshold", "pca", "fast_ica"]
            Available plugins, retrieved using `Preprocessors(category="dimensionality_reduction").list_available()`:
                - 'feature_agglomeration'
                - 'fast_ica'
                - 'variance_threshold'
                - 'gauss_projection'
                - 'pca'
                - 'nop' # no operation
        imputers: list.
            Plugin search pool to use in the pipeline for imputation. Defaults to ["mean", "ice", "missforest", "hyperimpute"].
            Available plugins, retrieved using `Imputers().list_available()`:
                - 'sinkhorn'
                - 'EM'
                - 'mice'
                - 'ice'
                - 'hyperimpute'
                - 'most_frequent'
                - 'median'
                - 'missforest'
                - 'softimpute'
                - 'nop'
                - 'mean'
                - 'gain'
        estimators: list.
            Plugin search pool to use in the pipeline for risk estimation. Defaults to ["survival_xgboost", "loglogistic_aft", "deephit", "cox_ph", "weibull_aft", "lognormal_aft", "coxnet"]
            Available plugins:
             - 'survival_xgboost'
             - 'loglogistic_aft'
             - 'deephit'
             - 'cox_ph'
             - 'weibull_aft'
             - 'lognormal_aft'
             - 'coxnet'
        hooks: Hooks.
            Custom callbacks to be notified about the search progress.
        random_state: int:
            Random seed
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        time_horizons: List[int],
        num_iter: int = 50,
        num_ensemble_iter: int = 100,
        timeout: int = 360,
        n_folds_cv: int = 3,
        estimators: List[str] = default_risk_estimation_names,
        ensemble_size: int = 2,
        imputers: List[str] = [],
        feature_scaling: List[str] = default_feature_scaling_names,
        feature_selection: List[str] = default_feature_selection_names,
        hooks: Hooks = DefaultHooks(),
        optimizer_type: str = "bayesian",
        random_state: int = 0,
    ) -> None:
        ensemble_size = min(ensemble_size, len(estimators))

        self.time_horizons = time_horizons
        self.num_ensemble_iter = num_ensemble_iter
        self.num_iter = num_iter
        self.timeout = timeout
        self.ensemble_size = ensemble_size
        self.n_folds_cv = n_folds_cv
        self.hooks = hooks

        self.study_name = study_name
        self.optimizer_type = optimizer_type
        self.random_state = random_state

        self.estimator_seeker = RiskEstimatorSeeker(
            study_name,
            time_horizons=time_horizons,
            num_iter=num_iter,
            n_folds_cv=n_folds_cv,
            top_k=ensemble_size,
            timeout=timeout,
            estimators=estimators,
            hooks=hooks,
            feature_scaling=feature_scaling,
            feature_selection=feature_selection,
            imputers=imputers,
            optimizer_type=optimizer_type,
            random_state=self.random_state,
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
                n_splits=self.n_folds_cv, shuffle=True, random_state=seed
            )
        else:
            skf = StratifiedKFold(
                n_splits=self.n_folds_cv, shuffle=True, random_state=seed
            )

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

            try:
                metrics = evaluate_survival_estimator(
                    cv_folds,
                    X,
                    T,
                    Y,
                    [time_horizon],
                    pretrained=True,
                    group_ids=group_ids,
                )
            except BaseException as e:
                log.error(f"evaluate_survival_ensemble failed: {e}")

                return 0

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
                f"Ensemble {cv_folds[0].name()} : results {metrics['raw']['c_index'][0]}"
            )
            return metrics["raw"]["c_index"][0] - metrics["raw"]["brier_score"][0]

        study = EnsembleOptimizer(
            study_name=f"{self.study_name}_risk_estimation_exploration_ensemble_{time_horizon}",
            ensemble_len=len(ensemble),
            evaluation_cbk=evaluate,
            optimizer_type=self.optimizer_type,
            n_trials=self.num_iter,
            timeout=self.timeout,
            skip_recap=skip_recap,
            random_state=self.random_state,
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
