# stdlib
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_feature_scaling_names,
    default_regressors_names,
)
from autoprognosis.explorers.regression_combos import RegressionEnsembleSeeker
from autoprognosis.hooks import Hooks
import autoprognosis.logger as log
from autoprognosis.studies._base import DefaultHooks, Study
from autoprognosis.studies._preprocessing import dataframe_hash, dataframe_preprocess
from autoprognosis.utils.serialization import load_model_from_file, save_model_to_file
from autoprognosis.utils.tester import evaluate_regression

PATIENCE = 10
SCORE_THRESHOLD = 0.65


class RegressionStudy(Study):
    """
    Core logic for regression studies.

    A study automatically handles imputation, preprocessing and model selection for a certain dataset.
    The output is an optimal model architecture, selected by the AutoML logic.

    Args:
        dataset: DataFrame.
            The dataset to analyze.
        target: str.
            The target column in the dataset.
        num_iter: int.
            Number of optimization iteration.
        num_study_iter: int.
            The number of study iterations.
        timeout: int.
            Max wait time for each estimator hyperparameter search.
        metric: str.
            The metric to use for optimization. ["r2"]
        study_name: str.
            The name of the study, to be used in the caches.
        feature_scaling: list.
            Plugins to use in the pipeline for preprocessing.
        regressors: list.
            Plugins to use in the pipeline for prediction.
        imputers: list.
            Plugins to use in the pipeline for imputation.
        hooks: Hooks.
            Custom callbacks to be notified about the search progress.
        workspace: Path.
            Where to store the output model.
        score_threshold: float.
            The minimum metric score for a candidate.
        id: str.
            The id column in the dataset.

    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str,
        num_iter: int = 20,
        num_study_iter: int = 5,
        timeout: int = 360,
        metric: str = "r2",
        study_name: Optional[str] = None,
        feature_scaling: List[str] = default_feature_scaling_names,
        regressors: List[str] = default_regressors_names,
        imputers: List[str] = ["ice"],
        workspace: Path = Path("tmp"),
        hooks: Hooks = DefaultHooks(),
        score_threshold: float = SCORE_THRESHOLD,
        nan_placeholder: Any = None,
        group_id: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.hooks = hooks
        dataset = pd.DataFrame(dataset)
        if nan_placeholder is not None:
            dataset = dataset.replace(nan_placeholder, np.nan)

        imputation_method: Optional[str] = None
        if dataset.isnull().values.any():
            if len(imputers) == 0:
                raise RuntimeError("Please provide at least one imputation method")

            if len(imputers) == 1:
                imputation_method = imputers[0]
                imputers = []
        else:
            imputers = []

        self.X, _, self.Y, _, _, self.group_ids = dataframe_preprocess(
            dataset, target, imputation_method=imputation_method, group_id=group_id
        )

        self.internal_name = dataframe_hash(dataset)
        self.study_name = study_name if study_name is not None else self.internal_name

        self.output_folder = Path(workspace) / self.study_name
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.output_file = self.output_folder / "model.p"

        self.num_study_iter = num_study_iter

        self.metric = metric
        self.score_threshold = score_threshold

        self.seeker = RegressionEnsembleSeeker(
            self.internal_name,
            num_iter=10,
            num_ensemble_iter=15,
            timeout=timeout,
            metric=metric,
            feature_scaling=feature_scaling,
            regressors=regressors,
            imputers=imputers,
            hooks=self.hooks,
        )

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Regression study search cancelled")

    def load_progress(self) -> Tuple[int, Any]:
        self._should_continue()

        if not self.output_file.is_file():
            return -1, None

        try:
            start = time.time()
            best_model = load_model_from_file(self.output_file)
            metrics = evaluate_regression(
                best_model, self.X, self.Y, group_ids=self.group_ids
            )
            best_score = metrics["clf"][self.metric][0]
            self.hooks.heartbeat(
                topic="regression_study",
                subtopic="candidate",
                event_type="candidate",
                name=best_model.name(),
                models=[mod.name() for mod in best_model.models],
                weights=best_model.weights,
                duration=time.time() - start,
                aucroc=metrics["str"]["r2"],
            )

            return best_score, best_model
        except BaseException:
            return -1, None

    def save_progress(self, model: Any) -> None:
        self._should_continue()

        if self.output_file:
            save_model_to_file(self.output_file, model)

    def run(self) -> Any:
        self._should_continue()

        best_score, best_model = self.load_progress()

        patience = 0
        for it in range(self.num_study_iter):
            self._should_continue()
            start = time.time()

            current_model = self.seeker.search(self.X, self.Y, group_ids=self.group_ids)

            metrics = evaluate_regression(
                current_model, self.X, self.Y, group_ids=self.group_ids
            )
            score = metrics["clf"][self.metric][0]

            self.hooks.heartbeat(
                topic="regression_study",
                subtopic="candidate",
                event_type="candidate",
                name=current_model.name(),
                duration=time.time() - start,
                aucroc=metrics["str"][self.metric],
            )

            if score < self.score_threshold:
                log.info(f"The ensemble is not good enough, keep searching {metrics}")
                continue

            if best_score >= score:
                log.info(
                    f"Model score not improved {score}. Previous best {best_score}"
                )
                patience += 1

                if patience > PATIENCE:
                    log.info(
                        f"Study not improved for {PATIENCE} iterations. Stopping..."
                    )
                    break
                continue

            patience = 0
            best_score = metrics["clf"][self.metric][0]
            best_model = current_model

            log.error(
                f"Best ensemble so far: {best_model.name()} with score {metrics['clf'][self.metric]}"
            )

            self.save_progress(best_model)

        return best_model
