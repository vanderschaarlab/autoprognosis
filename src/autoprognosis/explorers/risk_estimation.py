# stdlib
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

# third party
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pydantic import validate_arguments

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_risk_estimation_names,
)
from autoprognosis.explorers.core.optimizer import Optimizer
from autoprognosis.explorers.core.selector import PipelineSelector
from autoprognosis.explorers.hooks import DefaultHooks
from autoprognosis.hooks import Hooks
import autoprognosis.logger as log
from autoprognosis.utils.tester import evaluate_survival_estimator

dispatcher = Parallel(max_nbytes=None, backend="loky", n_jobs=2)


class RiskEstimatorSeeker:
    """
    AutoML core logic for risk estimation tasks.

    Args:
        study_name: str.
            Study ID, used for caching.
        time_horizons:list.
            list of time horizons.
        num_iter: int.
            Number of optimization trials.
        timeout: int.
            Max wait time(in seconds) for the optimization output.
        CV: int.
            Number of folds to use for evaluation
        top_k: int
            Number of candidates to return.
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
        timeout: int = 360,
        CV: int = 5,
        top_k: int = 1,
        estimators: List[str] = default_risk_estimation_names,
        feature_scaling: List[str] = default_feature_scaling_names,
        imputers: List[str] = [],
        hooks: Hooks = DefaultHooks(),
        optimizer_type: str = "bayesian",
        strict: bool = False,
    ) -> None:
        self.time_horizons = time_horizons

        self.num_iter = num_iter
        self.timeout = timeout
        self.top_k = top_k
        self.study_name = study_name
        self.hooks = hooks
        self.optimizer_type = optimizer_type
        self.strict = strict
        self.CV = CV

        self.estimators = [
            PipelineSelector(
                estimator,
                classifier_category="risk_estimation",
                calibration=[],
                feature_selection=[],
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
    ) -> Tuple[float, float, Dict]:
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

            self.hooks.heartbeat(
                topic="risk_estimation",
                subtopic="model_search",
                event_type="performance",
                name=model.name(),
                model_args=kwargs,
                duration=time.time() - start,
                horizon=time_horizon,
                aucroc=metrics["str"]["aucroc"],
                cindex=metrics["str"]["c_index"],
                brier_score=metrics["str"]["brier_score"],
            )
            return metrics["clf"]["c_index"][0] - metrics["clf"]["brier_score"][0]

        study = Optimizer(
            study_name=f"{self.study_name}_risk_estimation_exploration_{estimator.name()}_{time_horizon}",
            estimator=estimator,
            evaluation_cbk=evaluate_estimator,
            optimizer_type=self.optimizer_type,
            n_trials=self.num_iter,
            timeout=self.timeout,
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

        for idx, (best_score, best_args) in enumerate(search_results):
            all_scores.append([best_score])
            all_args.append([best_args])

            log.info(
                f"Time horizon {time_horizon}: evaluation for {self.estimators[idx].name()} scores:{best_score}. Args {best_args}"
            )

        all_scores_np = np.array(all_scores)
        selected_points = min(self.top_k, len(all_scores))
        best_scores = np.sort(np.unique(all_scores_np.ravel()))[-selected_points:]

        result = []
        for score in reversed(best_scores):
            pos = np.argwhere(all_scores_np == score)[0]
            pos_est = pos[0]
            est_args = pos[1]
            log.info(
                f"Selected score {score}: {self.estimators[pos_est].name()} : {all_args[pos_est][est_args]}"
            )
            result.append((pos_est, all_args[pos_est][est_args]))

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
            for idx, args in best_estimators_template:
                horizon_result.append(
                    self.estimators[idx].get_pipeline_from_named_args(**args)
                )
            result.append(horizon_result)

        return result
