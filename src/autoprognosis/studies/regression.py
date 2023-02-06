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
    default_feature_selection_names,
    default_regressors_names,
)
from autoprognosis.explorers.regression_combos import RegressionEnsembleSeeker
from autoprognosis.hooks import DefaultHooks, Hooks
import autoprognosis.logger as log
from autoprognosis.studies._base import Study
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.serialization import (
    dataframe_hash,
    load_model_from_file,
    save_model_to_file,
)
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
            Maximum Number of optimization trials. This is the limit of trials for each base estimator in the "regressors" list, used in combination with the "timeout" parameter. For each estimator, the search will end after "num_iter" trials or "timeout" seconds.
        num_study_iter: int.
            The number of study iterations. This is the limit for the outer optimization loop. After each outer loop, an intermediary model is cached and can be used by another process, while the outer loop continues to improve the result.
        timeout: int.
            Maximum wait time(seconds) for each estimator hyperparameter search. This timeout will apply to each estimator in the "regressors" list.
        metric: str.
            The metric to use for optimization.
            Available metric:
                - "r2"
        study_name: str.
            The name of the study, to be used in the caches.
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
        regressors: list.
            Plugin search pool to use in the pipeline for prediction. Defaults to ["random_forest_regressor","xgboost_regressor", "linear_regression", "catboost_regressor"]
            Available plugins, retrieved using `Regression().list_available()`:
                - 'kneighbors_regressor'
                - 'bayesian_ridge'
                - 'tabnet_regressor'
                - 'catboost_regressor'
                - 'random_forest_regressor'
                - 'mlp_regressor'
                - 'xgboost_regressor'
                - 'neural_nets_regression'
                - 'linear_regression'
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
        >>> import pandas as pd
        >>> from autoprognosis.utils.serialization import load_model_from_file
        >>> from autoprognosis.utils.tester import evaluate_regression
        >>> from autoprognosis.studies.regression import RegressionStudy
        >>>
        >>> # Load dataset
        >>> df = pd.read_csv(
        >>>     "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
        >>>     header=None,
        >>>     sep="\\t",
        >>> )
        >>> last_col = df.columns[-1]
        >>> y = df[last_col]
        >>> X = df.drop(columns=[last_col])
        >>>
        >>> df = X.copy()
        >>> df["target"] = y
        >>>
        >>> # Search the model
        >>>
        >>> study_name="regression_example"
        >>> study = RegressionStudy(
        >>>     study_name=study_name,
        >>>     dataset=df,  # pandas DataFrame
        >>>     target="target",  # the label column in the dataset
        >>> )
        >>> model = study.fit()
        >>>
        >>> # Predict using the model
        >>> model.predict(X)
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        target: str,
        num_iter: int = 20,
        num_study_iter: int = 5,
        num_ensemble_iter: int = 15,
        timeout: int = 360,
        metric: str = "r2",
        study_name: Optional[str] = None,
        feature_scaling: List[str] = default_feature_scaling_names,
        feature_selection: List[str] = default_feature_selection_names,
        regressors: List[str] = default_regressors_names,
        imputers: List[str] = ["ice"],
        workspace: Path = Path("tmp"),
        hooks: Hooks = DefaultHooks(),
        score_threshold: float = SCORE_THRESHOLD,
        nan_placeholder: Any = None,
        group_id: Optional[str] = None,
        random_state: int = 0,
        sample_for_search: bool = True,
        max_search_sample_size: int = 10000,
        ensemble_size: int = 3,
        n_folds_cv: int = 5,
    ) -> None:
        super().__init__()
        enable_reproducible_results(random_state)

        self.hooks = hooks
        dataset = pd.DataFrame(dataset)
        if nan_placeholder is not None:
            dataset = dataset.replace(nan_placeholder, np.nan)

        if dataset.isnull().values.any():
            if len(imputers) == 0:
                raise RuntimeError("Please provide at least one imputation method")
        else:
            imputers = []

        drop_cols = [target]
        self.group_ids = None
        if group_id is not None:
            drop_cols.append(group_id)
            self.group_ids = dataset[group_id]

        self.Y = dataset[target]
        self.X = dataset.drop(columns=drop_cols)
        self.n_folds_cv = n_folds_cv

        if sample_for_search:
            sample_size = min(len(self.Y), max_search_sample_size)

            self.search_Y = self.Y.sample(sample_size, random_state=random_state)
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
        self.random_state = random_state

        self.seeker = RegressionEnsembleSeeker(
            self.internal_name,
            num_iter=num_iter,
            num_ensemble_iter=num_ensemble_iter,
            timeout=timeout,
            metric=metric,
            feature_scaling=feature_scaling,
            feature_selection=feature_selection,
            regressors=regressors,
            imputers=imputers,
            hooks=self.hooks,
            random_state=self.random_state,
            n_folds_cv=n_folds_cv,
            ensemble_size=ensemble_size,
        )

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Regression study search cancelled")

    def _load_progress(self) -> Tuple[int, Any]:
        self._should_continue()

        if not self.output_file.is_file():
            return -1, None

        try:
            start = time.time()
            best_model = load_model_from_file(self.output_file)
            metrics = evaluate_regression(
                best_model,
                self.search_X,
                self.search_Y,
                group_ids=self.search_group_ids,
                n_folds=self.n_folds_cv,
            )
            best_score = metrics["raw"][self.metric][0]
            eval_metrics = {}
            for metric in metrics["raw"]:
                eval_metrics[metric] = metrics["raw"][metric][0]
                eval_metrics[f"{metric}_str"] = metrics["str"][metric]

            self.hooks.heartbeat(
                topic="regression_study",
                subtopic="candidate",
                event_type="candidate",
                name=best_model.name(),
                duration=time.time() - start,
                score=best_score,
                **eval_metrics,
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

            metrics = evaluate_regression(
                current_model,
                self.search_X,
                self.search_Y,
                group_ids=self.search_group_ids,
                n_folds=self.n_folds_cv,
            )
            score = metrics["raw"][self.metric][0]
            eval_metrics = {}
            for metric in metrics["raw"]:
                eval_metrics[metric] = metrics["raw"][metric][0]
                eval_metrics[f"{metric}_str"] = metrics["str"][metric]

            self.hooks.heartbeat(
                topic="regression_study",
                subtopic="candidate",
                event_type="candidate",
                name=current_model.name(),
                duration=time.time() - start,
                score=score,
                **eval_metrics,
            )

            if score < self.score_threshold:
                log.critical(
                    f"The ensemble is not good enough, keep searching {metrics['str']}"
                )
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
            best_score = metrics["raw"][self.metric][0]
            best_model = current_model

            log.error(
                f"Best ensemble so far: {best_model.name()} with score {metrics['raw'][self.metric]}"
            )

            self._save_progress(best_model)

        self.hooks.finish()

        if best_score < self.score_threshold:
            log.critical(
                f"Unable to find a model above threshold {self.score_threshold}. Returning None"
            )
            return None

        return best_model

    def fit(self) -> Any:
        """Run the study and train the model. The call returns the fitted model."""
        model = self.run()
        model.fit(self.X, self.Y)

        return model
