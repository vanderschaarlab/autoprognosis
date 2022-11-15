# stdlib
import copy
from typing import List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.model_selection import GroupKFold, KFold

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_regressors_names,
)
from autoprognosis.explorers.core.optimizer import EnsembleOptimizer
from autoprognosis.explorers.hooks import DefaultHooks
from autoprognosis.hooks import Hooks
import autoprognosis.logger as log
from autoprognosis.plugins.ensemble.regression import (
    BaseRegressionEnsemble,
    WeightedRegressionEnsemble,
)
from autoprognosis.utils.tester import evaluate_regression

# autoprognosis relative
from .regression import RegressionSeeker

EPS = 1e-8


class RegressionEnsembleSeeker:
    """
    AutoML core logic for regression ensemble search.

    Args:
        study_name: str.
            Study ID, used for caching keys.
        num_iter: int.
            Number of optimization trials.
        num_ensemble_iter: int.
            Number of optimization trials for the ensemble weights.
        timeout: int.
            Max wait time(in seconds) for the optimization output.
        CV: int.
            Number of folds to use for evaluation
        ensemble_size: int.
            Number of base models for the ensemble.
        metric: str.
            The metric to use for optimization. ["r2"]
        feature_scaling: list.
            Plugins to use in the pipeline for preprocessing.
        regressors: list.
            Plugins to use in the pipeline for prediction.
        imputers: list.
            Plugins to use in the pipeline for imputation.
        hooks: Hooks.
            Custom callbacks to be notified about the search progress.
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        num_iter: int = 100,
        num_ensemble_iter: int = 100,
        timeout: int = 360,
        CV: int = 5,
        ensemble_size: int = 3,
        metric: str = "r2",
        feature_scaling: List[str] = default_feature_scaling_names,
        regressors: List[str] = default_regressors_names,
        imputers: List[str] = [],
        hooks: Hooks = DefaultHooks(),
        optimizer_type: str = "bayesian",
    ) -> None:
        self.num_iter = num_ensemble_iter
        self.timeout = timeout
        self.ensemble_size = ensemble_size
        self.CV = CV
        self.metric = metric
        self.study_name = study_name
        self.hooks = hooks
        self.optimizer_type = optimizer_type

        self.seeker = RegressionSeeker(
            study_name,
            num_iter=num_iter,
            metric=metric,
            CV=CV,
            top_k=ensemble_size,
            timeout=timeout,
            feature_scaling=feature_scaling,
            regressors=regressors,
            hooks=hooks,
            imputers=imputers,
            optimizer_type=optimizer_type,
        )

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("regressor combo search cancelled")

    def pretrain_for_cv(
        self,
        ensemble: List,
        X: pd.DataFrame,
        Y: pd.Series,
        group_ids: Optional[pd.Series] = None,
        seed: int = 0,
    ) -> List:
        self._should_continue()

        if group_ids is not None:
            kf = GroupKFold(n_splits=self.CV)
        else:
            kf = KFold(n_splits=self.CV, shuffle=True, random_state=seed)

        folds = []
        for train_index, _ in kf.split(X, Y, groups=group_ids):
            X_train = X.loc[X.index[train_index]]
            Y_train = Y.loc[Y.index[train_index]]

            local_fold = []
            for estimator in ensemble:
                model = copy.deepcopy(estimator)
                model.fit(X_train, Y_train)
                local_fold.append(model)
            folds.append(local_fold)
        return folds

    def search_weights(
        self,
        ensemble: List,
        X: pd.DataFrame,
        Y: pd.Series,
        group_ids: Optional[pd.Series] = None,
    ) -> Tuple[WeightedRegressionEnsemble, float]:
        self._should_continue()

        pretrained_models = self.pretrain_for_cv(ensemble, X, Y, group_ids=group_ids)

        def evaluate(weights: List) -> float:
            self._should_continue()

            folds = []
            for fold in pretrained_models:
                folds.append(WeightedRegressionEnsemble(fold, weights))

            metrics = evaluate_regression(
                folds, X, Y, self.CV, pretrained=True, group_ids=group_ids
            )

            log.debug(f"ensemble {folds[0].name()} : results {metrics['clf']}")
            score = metrics["clf"][self.metric][0]

            return score

        study = EnsembleOptimizer(
            study_name=f"{self.study_name}_regressor_exploration_ensemble_v2",
            ensemble_len=len(ensemble),
            evaluation_cbk=evaluate,
            optimizer_type=self.optimizer_type,
            n_trials=self.num_iter,
            timeout=self.timeout,
        )

        best_score, selected_weights = study.evaluate()
        weights = []
        for idx in range(len(ensemble)):
            weights.append(selected_weights[f"weight_{idx}"])

        weights = weights / (np.sum(weights) + EPS)
        log.info(f"Best trial for ensemble: {best_score} for {weights}")

        return WeightedRegressionEnsemble(ensemble, weights), best_score

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def search(
        self, X: pd.DataFrame, Y: pd.Series, group_ids: Optional[pd.Series] = None
    ) -> BaseRegressionEnsemble:
        self._should_continue()

        best_models = self.seeker.search(X, Y, group_ids=group_ids)

        if self.hooks.cancel():
            raise StudyCancelled("regressor search cancelled")

        scores = []
        ensembles: list = []

        weighted_ensemble, weighted_ens_score = self.search_weights(
            best_models, X, Y, group_ids=group_ids
        )
        log.info(
            f"Weighted regression ensemble: {weighted_ensemble.name()} -> {weighted_ens_score}"
        )

        scores.append(weighted_ens_score)
        ensembles.append(weighted_ensemble)

        return ensembles[np.argmax(scores)]
