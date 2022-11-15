# stdlib
import time
from typing import Any, Dict, List, Optional, Tuple

# third party
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pydantic import validate_arguments

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_classifiers_names,
    default_feature_scaling_names,
)
from autoprognosis.explorers.core.optimizer import Optimizer
from autoprognosis.explorers.core.selector import PipelineSelector
from autoprognosis.explorers.hooks import DefaultHooks
from autoprognosis.hooks import Hooks
import autoprognosis.logger as log
from autoprognosis.utils.tester import evaluate_estimator

dispatcher = Parallel(max_nbytes=None, backend="loky", n_jobs=2)


class ClassifierSeeker:
    """
    AutoML core logic for classification tasks.

    Args:
        study_name: str.
            Study ID, used for caching.
        num_iter: int.
            Number of optimization trials.
        metric: str.
            The metric to use for optimization. ["aucroc", "aucprc"]
        CV: int.
            Number of folds to use for evaluation
        top_k: int
            Number of candidates to return
        timeout: int.
            Max wait time(in seconds) for the optimization output.
        feature_scaling: list.
            Plugins to use in the pipeline for preprocessing.
        classifiers: list.
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
        metric: str = "aucroc",
        CV: int = 5,
        top_k: int = 3,
        timeout: int = 360,
        feature_scaling: List[str] = default_feature_scaling_names,
        classifiers: List[str] = default_classifiers_names,
        imputers: List[str] = [],
        hooks: Hooks = DefaultHooks(),
        optimizer_type: str = "bayesian",
        strict: bool = False,
    ) -> None:
        for int_val in [num_iter, CV, top_k, timeout]:
            if int_val <= 0 or type(int_val) != int:
                raise ValueError(
                    f"invalid input number {int_val}. Should be a positive integer"
                )
        metrics = ["aucroc", "aucprc"]
        if metric not in metrics:
            raise ValueError(f"invalid input metric. Should be from {metrics}")

        self.study_name = study_name
        self.hooks = hooks

        self.estimators = [
            PipelineSelector(
                plugin,
                calibration=[],
                feature_scaling=feature_scaling,
                imputers=imputers,
            )
            for plugin in classifiers
        ]

        self.CV = CV
        self.num_iter = num_iter
        self.strict = strict
        self.timeout = timeout
        self.top_k = top_k
        self.metric = metric
        self.optimizer_type = optimizer_type

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Classifier search cancelled")

    def search_best_args_for_estimator(
        self,
        estimator: Any,
        X: pd.DataFrame,
        Y: pd.Series,
        group_ids: Optional[pd.Series] = None,
    ) -> Tuple[float, float, Dict]:
        self._should_continue()

        def evaluate_args(**kwargs: Any) -> float:
            self._should_continue()

            start = time.time()

            model = estimator.get_pipeline_from_named_args(**kwargs)
            try:
                metrics = evaluate_estimator(
                    model, X, Y, self.CV, metric=self.metric, group_ids=group_ids
                )
            except BaseException as e:
                log.error(f"evaluate_estimator failed: {e}")

                if self.strict:
                    raise

                return 0

            self.hooks.heartbeat(
                topic="classification",
                subtopic="model_search",
                event_type="performance",
                name=model.name(),
                model_args=kwargs,
                duration=time.time() - start,
                aucroc=metrics["str"][self.metric],
            )
            return metrics["clf"][self.metric][0]

        study = Optimizer(
            study_name=f"{self.study_name}_classifiers_exploration_{estimator.name()}",
            estimator=estimator,
            evaluation_cbk=evaluate_args,
            optimizer_type=self.optimizer_type,
            n_trials=self.num_iter,
            timeout=self.timeout,
        )
        return study.evaluate()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def search(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        group_ids: Optional[pd.Series] = None,
    ) -> List:
        """Search the optimal model for the task.

        Args:
            X: DataFrame
                The covariates
            y: DataFrame/Series
                The labels
            group_ids: Optional str
                Optional Group labels for the samples used while splitting the dataset into train/test set.

        """
        self._should_continue()

        search_results = dispatcher(
            delayed(self.search_best_args_for_estimator)(estimator, X, Y, group_ids)
            for estimator in self.estimators
        )

        all_scores = []
        all_args = []

        for idx, (best_score, best_args) in enumerate(search_results):
            all_scores.append([best_score])
            all_args.append([best_args])

            log.info(
                f"Evaluation for {self.estimators[idx].name()} scores: {best_score}. Args {best_args}"
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
            model = self.estimators[pos_est].get_pipeline_from_named_args(
                **all_args[pos_est][est_args]
            )
            result.append(model)

        return result
