# stdlib
import time
import traceback
from typing import Any, List, Optional, Tuple

# third party
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pydantic import validate_arguments

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_feature_selection_names,
    default_risk_estimation_names,
)
from autoprognosis.explorers.core.optimizer import Optimizer
from autoprognosis.explorers.core.selector import PipelineSelector
from autoprognosis.hooks import DefaultHooks, Hooks
import autoprognosis.logger as log
from autoprognosis.utils.parallel import n_opt_jobs
from autoprognosis.utils.tester import evaluate_survival_estimator

dispatcher = Parallel(max_nbytes=None, backend="loky", n_jobs=n_opt_jobs())


class RiskEstimatorSeeker:
    """
    AutoML core logic for risk estimation tasks.

    Args:
        study_name: str.
            Study ID, used for caching.
        time_horizons:list.
            list of time horizons.
        num_iter: int.
            Maximum Number of optimization trials. This is the limit of trials for each base estimator in the "risk_estimators" list, used in combination with the "timeout" parameter. For each estimator, the search will end after "num_iter" trials or "timeout" seconds.
        timeout: int.
            Maximum wait time(seconds) for each estimator hyperparameter search. This timeout will apply to each estimator in the "risk_estimators" list.
        n_folds_cv: int.
            Number of folds to use for evaluation
        top_k: int
            Number of candidates to return.
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
        estimators: list.
            Plugins to use in the pipeline for risk prediction.
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
        timeout: int = 360,
        n_folds_cv: int = 5,
        top_k: int = 1,
        estimators: List[str] = default_risk_estimation_names,
        feature_scaling: List[str] = default_feature_scaling_names,
        feature_selection: List[str] = default_feature_selection_names,
        imputers: List[str] = [],
        hooks: Hooks = DefaultHooks(),
        optimizer_type: str = "bayesian",
        strict: bool = False,
        random_state: int = 0,
    ) -> None:
        self.time_horizons = time_horizons

        self.num_iter = num_iter
        self.timeout = timeout
        self.top_k = top_k
        self.study_name = study_name
        self.hooks = hooks
        self.optimizer_type = optimizer_type
        self.strict = strict
        self.n_folds_cv = n_folds_cv
        self.random_state = random_state

        self.estimators = [
            PipelineSelector(
                estimator,
                classifier_category="risk_estimation",
                calibration=[],
                feature_selection=feature_selection,
                feature_scaling=feature_scaling,
                imputers=imputers,
            )
            for estimator in estimators
        ]

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("risk estimation search cancelled")

    def search_best_args_for_estimator(
        self,
        estimator: Any,
        X: pd.DataFrame,
        T: pd.DataFrame,
        Y: pd.DataFrame,
        time_horizon: int,
        group_ids: Optional[pd.Series] = None,
    ) -> Tuple[List[float], List[float]]:
        self._should_continue()

        def evaluate_estimator(**kwargs: Any) -> float:
            self._should_continue()
            start = time.time()
            time_horizons = [time_horizon]

            model = estimator.get_pipeline_from_named_args(**kwargs)

            try:
                metrics = evaluate_survival_estimator(
                    model, X, T, Y, time_horizons, group_ids=group_ids
                )
            except BaseException as e:
                log.error(f"evaluate_survival_estimator failed {e}")

                if self.strict:
                    raise

                return 0

            eval_metrics = {}
            for metric in metrics["raw"]:
                eval_metrics[metric] = metrics["raw"][metric][0]
                eval_metrics[f"{metric}_str"] = metrics["str"][metric]

            score = metrics["raw"]["c_index"][0] - metrics["raw"]["brier_score"][0]

            self.hooks.heartbeat(
                topic="risk_estimation",
                subtopic="model_search",
                event_type="performance",
                name=model.name(),
                model_args=kwargs,
                duration=time.time() - start,
                horizon=time_horizon,
                score=score,
                **eval_metrics,
            )
            return score

        study = Optimizer(
            study_name=f"{self.study_name}_risk_estimation_exploration_{estimator.name()}_{time_horizon}",
            estimator=estimator,
            evaluation_cbk=evaluate_estimator,
            optimizer_type=self.optimizer_type,
            n_trials=self.num_iter,
            timeout=self.timeout,
            random_state=self.random_state,
        )
        return study.evaluate()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def search_estimator(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        Y: pd.Series,
        time_horizon: int,
        group_ids: Optional[pd.Series] = None,
    ) -> List:
        self._should_continue()

        log.info(f"Searching estimators for horizon {time_horizon}")
        try:
            search_results = dispatcher(
                delayed(self.search_best_args_for_estimator)(
                    estimator, X, T, Y, time_horizon, group_ids=group_ids
                )
                for estimator in self.estimators
            )
        except BaseException as e:
            print(traceback.format_exc())
            raise e

        all_scores = []
        all_args = []
        all_estimators = []

        for idx, (best_scores, best_args) in enumerate(search_results):
            best_idx = np.argmax(best_scores)
            all_scores.append(best_scores[best_idx])
            all_args.append(best_args[best_idx])
            all_estimators.append(self.estimators[idx])

            log.info(
                f"Time horizon {time_horizon}: evaluation for {self.estimators[idx].name()} scores:{max(best_scores)}."
            )

        all_scores_np = np.array(all_scores)
        selected_points = min(self.top_k, len(all_scores))
        best_scores = np.sort(np.unique(all_scores_np.ravel()))[-selected_points:]

        result = []
        for score in reversed(best_scores):
            pos = np.argwhere(all_scores_np == score)[0]
            pos_est = pos[0]
            log.info(
                f"Selected score {score}: {all_estimators[pos_est].name()} : {all_args[pos_est]}"
            )
            result.append((all_estimators[pos_est], all_args[pos_est]))

        return result

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def search(
        self,
        X: pd.DataFrame,
        T: pd.Series,
        Y: pd.Series,
        group_ids: Optional[pd.Series] = None,
    ) -> List:
        self._should_continue()

        result = []
        for time_horizon in self.time_horizons:
            best_estimators_template = self.search_estimator(
                X, T, Y, time_horizon, group_ids=group_ids
            )
            horizon_result = []
            for est, args in best_estimators_template:
                horizon_result.append(est.get_pipeline_from_named_args(**args))
            result.append(horizon_result)

        return result
