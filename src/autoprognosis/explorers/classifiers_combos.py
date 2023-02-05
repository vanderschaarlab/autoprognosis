# stdlib
import copy
from typing import List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.core.defaults import (
    default_classifiers_names,
    default_feature_scaling_names,
    default_feature_selection_names,
)
from autoprognosis.explorers.core.optimizer import EnsembleOptimizer
from autoprognosis.hooks import DefaultHooks, Hooks
import autoprognosis.logger as log
from autoprognosis.plugins.ensemble.classifiers import (
    AggregatingEnsemble,
    BaseEnsemble,
    StackingEnsemble,
    WeightedEnsemble,
)
from autoprognosis.utils.tester import evaluate_estimator

# autoprognosis relative
from .classifiers import ClassifierSeeker

EPS = 1e-8


class EnsembleSeeker:
    """
    AutoML core logic for classification ensemble search.

    Args:
        study_name: str.
            Study ID, used for caching keys.
        num_iter: int.
            Maximum Number of optimization trials. This is the limit of trials for each base estimator in the "classifiers" list, used in combination with the "timeout" parameter. For each estimator, the search will end after "num_iter" trials or "timeout" seconds.
        num_ensemble_iter: int.
            Number of optimization trials for the ensemble weights.
        timeout: int.
            Maximum wait time(seconds) for each estimator hyperparameter search. This timeout will apply to each estimator in the "classifiers" list.
        n_folds_cv: int.
            Number of folds to use for evaluation
        ensemble_size: int.
            Number of base models for the ensemble.
        metric: str.
            The metric to use for optimization.
            Available objective metrics:
                - "aucroc" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
                - "aucprc" : The average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
                - "accuracy" : Accuracy classification score.
                - "f1_score_micro": F1 score is a harmonic mean of the precision and recall. This version uses the "micro" average: calculate metrics globally by counting the total true positives, false negatives and false positives.
                - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                - "f1_score_weighted": F1 score is a harmonic mean of the precision and recall. This version uses the "weighted" average: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
                - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
                - "kappa":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
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
        classifiers: list.
            Plugin search pool to use in the pipeline for prediction. Defaults to ["random_forest", "xgboost", "logistic_regression", "catboost"].
            Available plugins, retrieved using `Classifiers().list_available()`:
                - 'adaboost'
                - 'bernoulli_naive_bayes'
                - 'neural_nets'
                - 'linear_svm'
                - 'qda'
                - 'decision_trees'
                - 'logistic_regression'
                - 'hist_gradient_boosting'
                - 'extra_tree_classifier'
                - 'bagging'
                - 'gradient_boosting'
                - 'ridge_classifier'
                - 'gaussian_process'
                - 'perceptron'
                - 'lgbm'
                - 'catboost'
                - 'random_forest'
                - 'tabnet'
                - 'multinomial_naive_bayes'
                - 'lda'
                - 'gaussian_naive_bayes'
                - 'knn'
                - 'xgboost'
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
        hooks: Hooks.
            Custom callbacks to be notified about the search progress.
        random_state: int:
            Random seed
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        num_iter: int = 100,
        num_ensemble_iter: int = 100,
        timeout: int = 360,
        n_folds_cv: int = 5,
        ensemble_size: int = 3,
        metric: str = "aucroc",
        feature_scaling: List[str] = default_feature_scaling_names,
        feature_selection: List[str] = default_feature_selection_names,
        classifiers: List[str] = default_classifiers_names,
        imputers: List[str] = [],
        hooks: Hooks = DefaultHooks(),
        optimizer_type: str = "bayesian",
        random_state: int = 0,
    ) -> None:
        ensemble_size = min(ensemble_size, len(classifiers))

        self.num_iter = num_ensemble_iter
        self.timeout = timeout
        self.ensemble_size = ensemble_size
        self.n_folds_cv = n_folds_cv
        self.metric = metric
        self.study_name = study_name
        self.hooks = hooks
        self.optimizer_type = optimizer_type
        self.random_state = random_state

        self.seeker = ClassifierSeeker(
            study_name,
            num_iter=num_iter,
            metric=metric,
            n_folds_cv=n_folds_cv,
            top_k=ensemble_size,
            timeout=timeout,
            feature_scaling=feature_scaling,
            feature_selection=feature_selection,
            classifiers=classifiers,
            hooks=hooks,
            imputers=imputers,
            optimizer_type=optimizer_type,
            random_state=self.random_state,
        )

    def _should_continue(self) -> None:
        if self.hooks.cancel():
            raise StudyCancelled("Classifier combo search cancelled")

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
            skf = StratifiedGroupKFold(
                n_splits=self.n_folds_cv, shuffle=True, random_state=seed
            )
        else:
            skf = StratifiedKFold(
                n_splits=self.n_folds_cv, shuffle=True, random_state=seed
            )

        folds = []
        for train_index, _ in skf.split(X, Y, groups=group_ids):
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
    ) -> Tuple[WeightedEnsemble, float]:
        self._should_continue()

        pretrained_models = self.pretrain_for_cv(ensemble, X, Y, group_ids=group_ids)

        def evaluate(weights: List) -> float:
            self._should_continue()

            folds = []
            for fold in pretrained_models:
                folds.append(WeightedEnsemble(fold, weights))

            try:
                metrics = evaluate_estimator(
                    folds, X, Y, self.n_folds_cv, pretrained=True, group_ids=group_ids
                )
            except BaseException as e:
                log.error(f"evaluate_ensemble failed: {e}")

                return 0

            log.debug(f"ensemble {folds[0].name()} : results {metrics['raw']}")
            score = metrics["raw"][self.metric][0]

            return score

        study = EnsembleOptimizer(
            study_name=f"{self.study_name}_classifier_exploration_ensemble_{self.metric}",
            ensemble_len=len(ensemble),
            evaluation_cbk=evaluate,
            optimizer_type=self.optimizer_type,
            n_trials=self.num_iter,
            timeout=self.timeout,
            random_state=self.random_state,
        )

        best_score, selected_weights = study.evaluate()
        weights = []
        for idx in range(len(ensemble)):
            weights.append(selected_weights[f"weight_{idx}"])

        weights = weights / (np.sum(weights) + EPS)
        log.info(f"Best trial for ensemble: {best_score} for {weights}")

        return WeightedEnsemble(ensemble, weights), best_score

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def search(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        group_ids: Optional[pd.Series] = None,
    ) -> BaseEnsemble:
        self._should_continue()

        best_models = self.seeker.search(X, Y, group_ids=group_ids)

        if self.hooks.cancel():
            raise StudyCancelled("Classifier search cancelled")

        scores = []
        ensembles: list = []

        try:
            stacking_ensemble = StackingEnsemble(best_models, meta_model=best_models[0])
            stacking_ens_score = evaluate_estimator(
                stacking_ensemble, X, Y, self.n_folds_cv, group_ids=group_ids
            )["raw"][self.metric][0]
            log.info(
                f"Stacking ensemble: {stacking_ensemble.name()} --> {stacking_ens_score}"
            )
            scores.append(stacking_ens_score)
            ensembles.append(stacking_ensemble)
        except BaseException as e:
            log.info(f"StackingEnsemble failed {e}")

        if self.hooks.cancel():
            raise StudyCancelled("Classifier search cancelled")

        try:
            aggr_ensemble = AggregatingEnsemble(best_models)
            aggr_ens_score = evaluate_estimator(
                aggr_ensemble, X, Y, self.n_folds_cv, group_ids=group_ids
            )["raw"][self.metric][0]
            log.info(
                f"Aggregating ensemble: {aggr_ensemble.name()} --> {aggr_ens_score}"
            )

            scores.append(aggr_ens_score)
            ensembles.append(aggr_ensemble)
        except BaseException as e:
            log.info(f"AggregatingEnsemble failed {e}")

        if self.hooks.cancel():
            raise StudyCancelled("Classifier search cancelled")

        weighted_ensemble, weighted_ens_score = self.search_weights(
            best_models, X, Y, group_ids=group_ids
        )
        log.info(
            f"Weighted ensemble: {weighted_ensemble.name()} -> {weighted_ens_score}"
        )

        scores.append(weighted_ens_score)
        ensembles.append(weighted_ensemble)

        return ensembles[np.argmax(scores)]
