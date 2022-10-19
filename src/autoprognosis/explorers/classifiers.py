# stdlib
import time
from typing import Any, Dict, List, Tuple

# third party
from joblib import Parallel, delayed
import numpy as np
import optuna
import pandas as pd

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_classifiers_names,
    default_feature_scaling_names,
)
from autoprognosis.explorers.core.optimizer import EarlyStoppingExceeded, create_study
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
        self.CV = CV
        self.timeout = timeout
        self.top_k = top_k
        self.metric = metric

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Classifier search cancelled")

    def search_best_args_for_estimator(
        self,
        estimator: Any,
        X: pd.DataFrame,
        Y: pd.DataFrame,
    ) -> Tuple[float, float, Dict]:
        self._should_continue()

        def evaluate_args(**kwargs: Any) -> float:
            start = time.time()

            model = estimator.get_pipeline_from_named_args(**kwargs)

            try:
                metrics = evaluate_estimator(model, X, Y, self.CV, metric=self.metric)
            except BaseException as e:
                log.error(f"evaluate_estimator failed: {e}")
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

        baseline_score = evaluate_args()

        if len(estimator.hyperparameter_space()) == 0:
            return baseline_score, baseline_score, {}

        log.info(f"baseline score for {estimator.name()} {baseline_score}")

        study, pruner = create_study(
            study_name=f"{self.study_name}_classifiers_exploration_{estimator.name()}",
        )

        def objective(trial: optuna.Trial) -> float:
            self._should_continue()

            args = estimator.sample_hyperparameters(trial)
            pruner.check_trial(trial)

            score = evaluate_args(**args)

            pruner.report_score(score)

            return score

        try:
            study.optimize(objective, n_trials=self.num_iter, timeout=self.timeout)
        except EarlyStoppingExceeded:
            log.info("Early stopping triggered for search")

        log.info(
            f"Best trial for estimator {estimator.name()}: {study.best_value} for {study.best_trial.params}"
        )

        return baseline_score, study.best_value, study.best_trial.params

    def search(self, X: pd.DataFrame, Y: pd.DataFrame) -> List:
        self._should_continue()

        search_results = dispatcher(
            delayed(self.search_best_args_for_estimator)(estimator, X, Y)
            for estimator in self.estimators
        )

        all_scores = []
        all_args = []

        for idx, (baseline_score, best_score, best_args) in enumerate(search_results):
            all_scores.append([baseline_score, best_score])
            all_args.append([{}, best_args])

            log.info(
                f"Evaluation for {self.estimators[idx].name()} scores: baseline {baseline_score} optimized {best_score}. Args {best_args}"
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
