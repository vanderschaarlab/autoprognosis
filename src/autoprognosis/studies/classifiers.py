# stdlib
from pathlib import Path
import time
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.classifiers_combos import EnsembleSeeker
from autoprognosis.explorers.core.defaults import (
    default_classifiers_names,
    default_feature_scaling_names,
    default_feature_selection_names,
)
from autoprognosis.hooks import Hooks
import autoprognosis.logger as log
from autoprognosis.studies._base import DefaultHooks, Study
from autoprognosis.studies._preprocessing import dataframe_hash, dataframe_preprocess
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.serialization import load_model_from_file, save_model_to_file
from autoprognosis.utils.tester import evaluate_estimator

PATIENCE = 10
SCORE_THRESHOLD = 0.65


class ClassifierStudy(Study):
    """
    Core logic for classification studies.

    A study automatically handles imputation, preprocessing and model selection for a certain dataset.
    The output is an optimal model architecture, selected by the AutoML logic.

    Args:
        dataset: DataFrame.
            The dataset to analyze.
        target: str.
            The target column in the dataset.
        num_iter: int.
            Number of optimization iterations. This is the limit of trials for each base model, e.g. xgboost.
        num_study_iter: int.
            The number of study iterations. This is the limit for the outer optimization loop. After each outer loop, an intermediary model is cached and can be used by another process, while the outer loop continues to improve the result.
        timeout: int.
            Max wait time for each estimator hyperparameter search.
        metric: str.
            The metric to use for optimization. ["aucroc", "aucprc"]
        study_name: str.
            The name of the study, to be used in the caches.
        feature_scaling: list.
            Plugins to use in the pipeline for scaling.
        feature_selection: list.
            Plugins to use in the pipeline for feature selection.
        classifiers: list.
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
        random_state: int
            Random seed
        sample_for_search: bool
            Subsample the evaluation dataset in the search pipeline. Improves the speed of the search.
        max_search_sample_size: int
            Subsample size for the evaluation dataset, if `sample` is True.
    Example:
        >>> from sklearn.datasets import load_breast_cancer
        >>>
        >>> from autoprognosis.studies.classifiers import ClassifierStudy
        >>> from autoprognosis.utils.serialization import load_model_from_file
        >>> from autoprognosis.utils.tester import evaluate_estimator
        >>>
        >>> X, Y = load_breast_cancer(return_X_y=True, as_frame=True)
        >>>
        >>> df = X.copy()
        >>> df["target"] = Y
        >>>
        >>> study_name = "example"
        >>>
        >>> study = ClassifierStudy(
        >>>     study_name=study_name,
        >>>     dataset=df,  # pandas DataFrame
        >>>     target="target",  # the label column in the dataset
        >>> )
        >>> model = study.fit()
        >>>
        >>> # Predict the probabilities of each class using the model
        >>> model.predict_proba(X)
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str,
        num_iter: int = 20,
        num_study_iter: int = 5,
        timeout: int = 360,
        metric: str = "aucroc",
        study_name: Optional[str] = None,
        feature_scaling: List[str] = default_feature_scaling_names,
        feature_selection: List[str] = default_feature_selection_names,
        classifiers: List[str] = default_classifiers_names,
        imputers: List[str] = ["ice"],
        workspace: Path = Path("tmp"),
        hooks: Hooks = DefaultHooks(),
        score_threshold: float = SCORE_THRESHOLD,
        group_id: Optional[str] = None,
        nan_placeholder: Any = None,
        random_state: int = 0,
        sample_for_search: bool = True,
        max_search_sample_size: int = 10000,
    ) -> None:
        super().__init__()
        enable_reproducible_results(random_state)

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
        else:
            imputers = []

        self.X, _, self.Y, _, _, self.group_ids = dataframe_preprocess(
            dataset,
            target,
            imputation_method=imputation_method,
            group_id=group_id,
        )

        if sample_for_search:
            sample_size = min(len(self.Y), max_search_sample_size)

            counts = self.Y.value_counts().to_dict()
            weights = self.Y.apply(lambda s: counts[s])
            self.search_Y = self.Y.sample(
                sample_size, random_state=random_state, weights=weights
            )
            self.search_X = self.X.loc[self.search_Y.index].copy()
            self.search_group_ids = None
            if self.group_ids:
                self.search_group_ids = self.group_ids.loc[self.search_Y.index].copy()
        else:
            self.search_X = self.X
            self.search_Y = self.Y
            self.search_group_ids = self.group_ids

        self.internal_name = dataframe_hash(dataset)
        self.study_name = study_name if study_name is not None else self.internal_name

        self.output_folder = Path(workspace) / self.study_name
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.output_file = self.output_folder / "model.p"

        self.num_study_iter = num_study_iter

        self.metric = metric
        self.score_threshold = score_threshold

        self.seeker = EnsembleSeeker(
            self.internal_name,
            num_iter=10,
            num_ensemble_iter=15,
            timeout=timeout,
            metric=metric,
            feature_scaling=feature_scaling,
            feature_selection=feature_selection,
            classifiers=classifiers,
            imputers=imputers,
            hooks=self.hooks,
        )

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Classifier study search cancelled")

    def _load_progress(self) -> Tuple[int, Any]:
        self._should_continue()

        if not self.output_file.is_file():
            return -1, None

        try:
            start = time.time()
            best_model = load_model_from_file(self.output_file)
            metrics = evaluate_estimator(
                best_model,
                self.search_X,
                self.search_Y,
                metric=self.metric,
                group_ids=self.search_group_ids,
            )
            best_score = metrics["clf"][self.metric][0]
            self.hooks.heartbeat(
                topic="classification_study",
                subtopic="candidate",
                event_type="candidate",
                name=best_model.name(),
                models=[mod.name() for mod in best_model.models],
                weights=best_model.weights,
                duration=time.time() - start,
                aucroc=metrics["str"]["aucroc"],
            )

            return best_score, best_model
        except BaseException:
            return -1, None

    def _save_progress(self, model: Any) -> None:
        self._should_continue()

        if self.output_file:
            save_model_to_file(self.output_file, model)

    def run(self) -> Any:
        """Run the study. The call returns the optimal model architecture - not fitted."""
        self._should_continue()

        best_score, best_model = self._load_progress()

        patience = 0
        for it in range(self.num_study_iter):
            self._should_continue()
            start = time.time()

            current_model = self.seeker.search(
                self.search_X, self.search_Y, group_ids=self.search_group_ids
            )

            metrics = evaluate_estimator(
                current_model,
                self.search_X,
                self.search_Y,
                metric=self.metric,
                group_ids=self.search_group_ids,
            )
            score = metrics["clf"][self.metric][0]

            self.hooks.heartbeat(
                topic="classification_study",
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

            self._save_progress(best_model)

        return best_model

    def fit(self) -> Any:
        """Run the study and train the model. The call returns the fitted model."""
        model = self.run()
        model.fit(self.X, self.Y)

        return model
