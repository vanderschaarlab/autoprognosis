# stdlib
import copy
from typing import Any, Dict, List, Optional, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
import autoprognosis.logger as log
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.metrics import (
    evaluate_auc,
    evaluate_brier_score,
    evaluate_c_index,
    generate_score,
    print_score,
)
from autoprognosis.utils.risk_estimation import generate_dataset_for_horizon

clf_supported_metrics = [
    "aucroc",
    "aucprc",
    "accuracy",
    "f1_score_micro",
    "f1_score_macro",
    "f1_score_weighted",
    "kappa",
    "kappa_quadratic",
    "precision_micro",
    "precision_macro",
    "precision_weighted",
    "recall_micro",
    "recall_macro",
    "recall_weighted",
    "mcc",
]
survival_supported_metrics = [
    "c_index",
    "brier_score",
    "aucroc",
    "sensitivity",
    "specificity",
    "PPV",
    "NPV",
    "predicted_cases",
]
reg_supported_metrics = ["mse", "mae", "r2"]


class classifier_metrics:
    """Helper class for evaluating the performance of the classifier.

    Args:
        metric: list, default=["aucroc", "aucprc", "accuracy", "f1_score_micro", "f1_score_macro", "f1_score_weighted",  "kappa", "precision_micro", "precision_macro", "precision_weighted", "recall_micro", "recall_macro", "recall_weighted",  "mcc",]
            The type of metric to use for evaluation.
            Potential values:
                - "aucroc" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
                - "aucprc" : The average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
                - "accuracy" : Accuracy classification score.
                - "f1_score_micro": F1 score is a harmonic mean of the precision and recall. This version uses the "micro" average: calculate metrics globally by counting the total true positives, false negatives and false positives.
                - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                - "f1_score_weighted": F1 score is a harmonic mean of the precision and recall. This version uses the "weighted" average: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
                - "kappa", "kappa_quadratic":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
                - "precision_micro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(micro) calculates metrics globally by counting the total true positives.
                - "precision_macro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
                - "precision_weighted": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
                - "recall_micro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(micro) calculates metrics globally by counting the total true positives.
                - "recall_macro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
                - "recall_weighted": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
                - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
    """

    def __init__(self, metric: Union[str, list] = clf_supported_metrics) -> None:
        if isinstance(metric, str):
            self.metrics = [metric]
        else:
            self.metrics = metric

    def get_metric(self) -> Union[str, list]:
        return self.metrics

    def score_proba(
        self, y_test: np.ndarray, y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        if y_test is None or y_pred_proba is None:
            raise RuntimeError("Invalid input for score_proba")

        results = {}
        y_pred = np.argmax(np.asarray(y_pred_proba), axis=1)
        for metric in self.metrics:
            if metric == "aucprc":
                results[metric] = self.average_precision_score(y_test, y_pred_proba)
            elif metric == "aucroc":
                results[metric] = self.roc_auc_score(y_test, y_pred_proba)
            elif metric == "accuracy":
                results[metric] = accuracy_score(y_test, y_pred)
            elif metric == "f1_score_micro":
                results[metric] = f1_score(
                    y_test, y_pred, average="micro", zero_division=0
                )
            elif metric == "f1_score_macro":
                results[metric] = f1_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
            elif metric == "f1_score_weighted":
                results[metric] = f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
            elif metric == "kappa":
                results[metric] = cohen_kappa_score(y_test, y_pred)
            elif metric == "kappa_quadratic":
                results[metric] = cohen_kappa_score(y_test, y_pred, weights="quadratic")
            elif metric == "recall_micro":
                results[metric] = recall_score(
                    y_test, y_pred, average="micro", zero_division=0
                )
            elif metric == "recall_macro":
                results[metric] = recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
            elif metric == "recall_weighted":
                results[metric] = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
            elif metric == "precision_micro":
                results[metric] = precision_score(
                    y_test, y_pred, average="micro", zero_division=0
                )
            elif metric == "precision_macro":
                results[metric] = precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
            elif metric == "precision_weighted":
                results[metric] = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
            elif metric == "mcc":
                results[metric] = matthews_corrcoef(y_test, y_pred)
            else:
                raise ValueError(f"invalid metric {metric}")

        log.debug(f"evaluate_classifier: {results}")
        return results

    def roc_auc_score(self, y_test: np.ndarray, y_pred_proba: np.ndarray) -> float:
        return evaluate_auc(y_test, y_pred_proba)[0]

    def average_precision_score(
        self, y_test: np.ndarray, y_pred_proba: np.ndarray
    ) -> float:

        return evaluate_auc(y_test, y_pred_proba)[1]


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_estimator(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    Y: Union[pd.Series, np.ndarray, List],
    n_folds: int = 3,
    seed: int = 0,
    pretrained: bool = False,
    group_ids: Optional[pd.Series] = None,
    *args: Any,
    **kwargs: Any,
) -> Dict:
    """Helper for evaluating classifiers.

    Args:
        estimator:
            Baseline model to evaluate. if pretrained == False, it must not be fitted.
        X: pd.DataFrame or np.ndarray:
            The covariates
        Y: pd.Series or np.ndarray or list:
            The labels
        n_folds: int
            cross-validation folds
        seed: int
            Random seed
        pretrained: bool
            If the estimator was already trained or not.
        group_ids: pd.Series
            The group_ids to use for stratified cross-validation

    Returns:
        Dict containing "raw" and "str" nodes. The "str" node contains prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric.
        Both "raw" and "str" nodes contain the following metrics:
            - "aucroc" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
            - "aucprc" : The average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
            - "accuracy" : Accuracy classification score.
            - "f1_score_micro": F1 score is a harmonic mean of the precision and recall. This version uses the "micro" average: calculate metrics globally by counting the total true positives, false negatives and false positives.
            - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            - "f1_score_weighted": F1 score is a harmonic mean of the precision and recall. This version uses the "weighted" average: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
            - "kappa":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
            - "precision_micro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(micro) calculates metrics globally by counting the total true positives.
            - "precision_macro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
            - "precision_weighted": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
            - "recall_micro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(micro) calculates metrics globally by counting the total true positives.
            - "recall_macro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
            - "recall_weighted": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
            - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.

    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    enable_reproducible_results(seed)

    X = pd.DataFrame(X).reset_index(drop=True)
    Y = LabelEncoder().fit_transform(Y)
    Y = pd.Series(Y).reset_index(drop=True)
    if group_ids is not None:
        group_ids = pd.Series(group_ids).reset_index(drop=True)

    log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    results = {}

    evaluator = classifier_metrics()
    for metric in clf_supported_metrics:
        results[metric] = np.zeros(n_folds)

    indx = 0
    if group_ids is not None:
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # group_ids is always ignored for StratifiedKFold so safe to pass None
    for train_index, test_index in skf.split(X, Y, groups=group_ids):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        if pretrained:
            model = estimator[indx]
        else:
            model = copy.deepcopy(estimator)
            model.fit(X_train, Y_train)

        preds = model.predict_proba(X_test)

        scores = evaluator.score_proba(Y_test, preds)
        for metric in scores:
            results[metric][indx] = scores[metric]

        indx += 1

    output_clf = {}
    output_clf_str = {}

    for key in results:
        key_out = generate_score(results[key])
        output_clf[key] = key_out
        output_clf_str[key] = print_score(key_out)

    return {
        "raw": output_clf,
        "str": output_clf_str,
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_estimator_multiple_seeds(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    Y: Union[pd.Series, np.ndarray, List],
    n_folds: int = 3,
    seeds: List[int] = [0, 1, 2],
    pretrained: bool = False,
    group_ids: Optional[pd.Series] = None,
) -> Dict:
    """Helper for evaluating classifiers with multiple seeds.

    Args:
        estimator:
            Baseline model to evaluate. if pretrained == False, it must not be fitted.
        X: pd.DataFrame or np.ndarray:
            The covariates
        Y: pd.Series or np.ndarray or list:
            The labels
        n_folds: int
            cross-validation folds
        seeds: List
            Random seeds
        pretrained: bool
            If the estimator was already trained or not.
        group_ids: pd.Series
            The group_ids to use for stratified cross-validation

    Returns:
        Dict containing "seeds", "agg" and "str" nodes. The "str" node contains the aggregated prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric. The "seeds" node contains the results for each random seed.
        Both "agg" and "str" nodes contain the following metrics:
            - "aucroc" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
            - "aucprc" : The average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
            - "accuracy" : Accuracy classification score.
            - "f1_score_micro": F1 score is a harmonic mean of the precision and recall. This version uses the "micro" average: calculate metrics globally by counting the total true positives, false negatives and false positives.
            - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            - "f1_score_weighted": F1 score is a harmonic mean of the precision and recall. This version uses the "weighted" average: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
            - "kappa":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
            - "precision_micro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(micro) calculates metrics globally by counting the total true positives.
            - "precision_macro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
            - "precision_weighted": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
            - "recall_micro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(micro) calculates metrics globally by counting the total true positives.
            - "recall_macro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
            - "recall_weighted": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
            - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.


    """
    results = {
        "seeds": {},
        "agg": {},
        "str": {},
    }

    repeats = {}
    for metric in clf_supported_metrics:
        repeats[metric] = []

    for seed in seeds:
        score = evaluate_estimator(
            estimator,
            X=X,
            Y=Y,
            n_folds=n_folds,
            seed=seed,
            pretrained=pretrained,
            group_ids=group_ids,
        )

        results["seeds"][seed] = score["str"]
        for metric in score["raw"]:
            repeats[metric].append(score["raw"][metric][0])

    for metric in repeats:
        output_clf = generate_score(repeats[metric])
        results["agg"][metric] = output_clf
        results["str"][metric] = print_score(output_clf)

    return results


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_survival_estimator(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    T: Union[pd.Series, np.ndarray, List],
    Y: Union[pd.Series, np.ndarray, List],
    time_horizons: Union[List[float], np.ndarray],
    n_folds: int = 3,
    seed: int = 0,
    pretrained: bool = False,
    risk_threshold: float = 0.5,
    group_ids: Optional[pd.Series] = None,
) -> Dict:
    """Helper for evaluating survival analysis tasks.

    Args:
        estimator:
            Baseline model to evaluate. if pretrained == False, it must not be fitted.
        X: DataFrame or np.ndarray
            The covariates
        T: Series or np.ndarray or list
            time to event/censoring values
        Y: Series or np.ndarray or list
            event or censored
        time_horizons: list or np.ndarray
            Horizons where to evaluate the performance.
        n_folds: int
            Number of folds for cross validation
        seed: int
            Random seed
        pretrained: bool
            If the estimator was trained or not
        group_ids:
            Group labels for the samples used while splitting the dataset into train/test set.

    Returns:
        Dict containing "raw", "str" and "horizons" nodes. The "str" node contains prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric. The "horizons" node splits the metrics by horizon.
        Each nodes contain the following metrics:
            - "c_index" : The concordance index or c-index is a metric to evaluate the predictions made by a survival algorithm. It is defined as the proportion of concordant pairs divided by the total number of possible evaluation pairs.
            - "brier_score": The Brier Score is a strictly proper score function or strictly proper scoring rule that measures the accuracy of probabilistic predictions.
            - "aucroc" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
            - "sensitivity": Sensitivity (true positive rate) is the probability of a positive test result, conditioned on the individual truly being positive.
            - "specificity": Specificity (true negative rate) is the probability of a negative test result, conditioned on the individual truly being negative.
            - "PPV": The positive predictive value(PPV) is the probability that following a positive test result, that individual will truly have that specific disease.
            - "NPV": The negative predictive value(NPV) is the probability that following a negative test result, that individual will truly not have that specific disease.

    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    enable_reproducible_results(seed)

    X = pd.DataFrame(X).reset_index(drop=True)
    Y = pd.Series(Y).reset_index(drop=True)
    T = pd.Series(T).reset_index(drop=True)
    if group_ids is not None:
        group_ids = pd.Series(group_ids).reset_index(drop=True)

    results = {}
    for metric in survival_supported_metrics:
        results[metric] = {}
        for horizon in time_horizons:
            results[metric][horizon] = np.zeros(n_folds)

    def _get_surv_metrics(
        cv_idx: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        T_train: pd.DataFrame,
        T_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
        time_horizons: list,
    ) -> tuple:
        train_max = T_train.max()
        T_test[T_test > train_max] = train_max

        if pretrained:
            model = estimator[cv_idx]
        else:
            model = copy.deepcopy(estimator)

            constant_cols = _constant_columns(X_train)
            X_train = X_train.drop(columns=constant_cols)
            X_test = X_test.drop(columns=constant_cols)

            model.fit(X_train, T_train, Y_train)

        pred = model.predict(X_test, time_horizons).to_numpy()

        c_index = []
        brier_score = []

        for k in range(len(time_horizons)):
            eval_horizon = min(time_horizons[k], np.max(T_test) - 1)
            c_index.append(
                evaluate_c_index(
                    T_train, Y_train, pred[:, k], T_test, Y_test, eval_horizon
                )
            )
            brier_score.append(
                evaluate_brier_score(
                    T_train, Y_train, pred[:, k], T_test, Y_test, eval_horizon
                )
            )

        return {
            "c_index": c_index,
            "brier_score": brier_score,
        }

    def _get_clf_metrics(
        cv_idx: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        T_train: pd.DataFrame,
        T_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
        hidx: int,
    ) -> Dict[str, float]:
        train_max = T_train.max()
        T_test[T_test > train_max] = train_max

        if pretrained:
            model = estimator[cv_idx]
        else:
            model = copy.deepcopy(estimator)

            constant_cols = _constant_columns(X_train)
            X_train = X_train.drop(columns=constant_cols)
            X_test = X_test.drop(columns=constant_cols)

            model.fit(X_train, T_train, Y_train)

        pred = model.predict(X_test, time_horizons).to_numpy()

        local_scores = pd.DataFrame(pred[:, hidx]).squeeze()
        local_preds = (local_scores > risk_threshold).astype(int)

        output = {
            "aucroc": roc_auc_score(Y_test, local_scores),
            "specificity": recall_score(
                Y_test, local_preds, pos_label=0, zero_division=0
            ),
            "sensitivity": recall_score(
                Y_test, local_preds, pos_label=1, zero_division=0
            ),
            "PPV": precision_score(Y_test, local_preds, pos_label=1, zero_division=0),
            "NPV": precision_score(Y_test, local_preds, pos_label=0, zero_division=0),
            "predicted_cases": local_preds.sum(),
        }
        return output

    if group_ids is not None:
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    cv_idx = 0
    for train_index, test_index in skf.split(X, Y, groups=group_ids):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        T_train = T.loc[T.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]
        T_test = T.loc[T.index[test_index]]

        local_time_horizons = [t for t in time_horizons if t > np.min(T_test)]

        local_surv_metrics = _get_surv_metrics(
            cv_idx,
            X_train,
            X_test,
            T_train,
            T_test,
            Y_train,
            Y_test,
            local_time_horizons,
        )
        for metric in local_surv_metrics:
            for hidx, horizon in enumerate(local_time_horizons):
                results[metric][horizon][cv_idx] = local_surv_metrics[metric][hidx]

        cv_idx += 1

    for k in range(len(time_horizons)):
        cv_idx = 0

        X_horizon, T_horizon, Y_horizon = generate_dataset_for_horizon(
            X, T, Y, time_horizons[k]
        )
        for train_index, test_index in skf.split(
            X_horizon, Y_horizon, groups=group_ids
        ):

            X_train = X_horizon.loc[X_horizon.index[train_index]]
            Y_train = Y_horizon.loc[Y_horizon.index[train_index]]
            T_train = T_horizon.loc[T_horizon.index[train_index]]
            X_test = X_horizon.loc[X_horizon.index[test_index]]
            Y_test = Y_horizon.loc[Y_horizon.index[test_index]]
            T_test = T_horizon.loc[T_horizon.index[test_index]]

            clf_metrics = _get_clf_metrics(
                cv_idx,
                X_train,
                X_test,
                T_train,
                T_test,
                Y_train,
                Y_test,
                k,
            )
            for metric in clf_metrics:
                results[metric][time_horizons[k]][cv_idx] = clf_metrics[metric]

            cv_idx += 1

    output: dict = {
        "horizons": {
            "raw": {},
            "str": {},
        },
        "str": {},
        "raw": {},
    }
    for metric in results:
        local_values = []
        for horizon in time_horizons:
            local_score = generate_score(results[metric][horizon])
            local_values.append(local_score[0])
            if horizon not in output["horizons"]["raw"]:
                output["horizons"]["raw"][horizon] = {}
                output["horizons"]["str"][horizon] = {}
            output["horizons"]["raw"][horizon][metric] = local_score
            output["horizons"]["str"][horizon][metric] = print_score(local_score)

        output["raw"][metric] = generate_score(local_values)
        output["str"][metric] = print_score(output["raw"][metric])

    return output


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_survival_estimator_multiple_seeds(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    T: Union[pd.Series, np.ndarray, List],
    Y: Union[pd.Series, np.ndarray, List],
    time_horizons: Union[List[float], np.ndarray],
    n_folds: int = 3,
    pretrained: bool = False,
    risk_threshold: float = 0.5,
    group_ids: Optional[pd.Series] = None,
    seeds: List[int] = [0, 1, 2],
) -> Dict:
    """Helper for evaluating survival analysis tasks with multiple random seeds.

    Args:
        estimator:
            Baseline model to evaluate. if pretrained == False, it must not be fitted.
        X: DataFrame or np.ndarray
            The covariates
        T: Series or np.ndarray or list
            time to event
        Y: Series or np.ndarray or list
            event or censored
        time_horizons: list or np.ndarray
            Horizons where to evaluate the performance.
        n_folds: int
            Number of folds for cross validation
        seeds: List
            Random seeds
        pretrained: bool
            If the estimator was trained or not
        group_ids:
            Group labels for the samples used while splitting the dataset into train/test set.

    Returns:
        Dict containing "seeds", "agg" and "str" nodes. The "str" node contains the aggregated prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric. The "seeds" node contains the results for each random seed.
        Both "agg" and "str" nodes contain the following metrics:
            - "c_index" : The concordance index or c-index is a metric to evaluate the predictions made by a survival algorithm. It is defined as the proportion of concordant pairs divided by the total number of possible evaluation pairs.
            - "brier_score": The Brier Score is a strictly proper score function or strictly proper scoring rule that measures the accuracy of probabilistic predictions.
            - "aucroc" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
            - "sensitivity": Sensitivity (true positive rate) is the probability of a positive test result, conditioned on the individual truly being positive.
            - "specificity": Specificity (true negative rate) is the probability of a negative test result, conditioned on the individual truly being negative.
            - "PPV": The positive predictive value(PPV) is the probability that following a positive test result, that individual will truly have that specific disease.
            - "NPV": The negative predictive value(NPV) is the probability that following a negative test result, that individual will truly not have that specific disease.
    """

    metrics = survival_supported_metrics
    results = {
        "seeds": {},
        "agg": {},
        "str": {},
    }

    repeats = {}
    for metric in metrics:
        repeats[metric] = []
    for seed in seeds:
        score = evaluate_survival_estimator(
            estimator,
            X=X,
            T=T,
            Y=Y,
            time_horizons=time_horizons,
            n_folds=n_folds,
            risk_threshold=risk_threshold,
            seed=seed,
            pretrained=pretrained,
            group_ids=group_ids,
        )

        results["seeds"][seed] = score["str"]
        for metric in metrics:
            repeats[metric].append(score["raw"][metric][0])

    for metric in metrics:
        output_clf = generate_score(repeats[metric])
        results["agg"][metric] = output_clf
        results["str"][metric] = print_score(output_clf)

    return results


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_regression(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    Y: Union[pd.Series, np.ndarray, List],
    n_folds: int = 3,
    seed: int = 0,
    pretrained: bool = False,
    group_ids: Optional[pd.Series] = None,
    *args: Any,
    **kwargs: Any,
) -> Dict:
    """Helper for evaluating regression tasks.

    Args:
        estimator:
            Baseline model to evaluate. if pretrained == False, it must not be fitted.
        X: pd.DataFrame or np.ndarray
            covariates
        Y: pd.Series or np.ndarray or list
            outcomes
        n_folds: int
            Number of cross-validation folds
        seed: int
            Random seed
        group_ids: pd.Series
            Optional group_ids for stratified cross-validation

    Returns:
        Dict containing "raw" and "str" nodes. The "str" node contains prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric.
        Both "raw" and "str" nodes contain the following metrics:
            - "r2": R^2(coefficient of determination) regression score function.
            - "mse": Mean squared error regression loss.
            - "mae": Mean absolute error regression loss.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    enable_reproducible_results(seed)
    metrics = reg_supported_metrics

    X = pd.DataFrame(X).reset_index(drop=True)
    Y = pd.Series(Y).reset_index(drop=True)
    if group_ids is not None:
        group_ids = pd.Series(group_ids).reset_index(drop=True)

    log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    metrics_ = {}
    for metric in metrics:
        metrics_[metric] = np.zeros(n_folds)

    indx = 0
    if group_ids is not None:
        kf = GroupKFold(n_splits=n_folds)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in kf.split(X, Y, groups=group_ids):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        if pretrained:
            model = estimator[indx]
        else:
            model = copy.deepcopy(estimator)
            model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        metrics_["mse"][indx] = mean_squared_error(Y_test, preds)
        metrics_["mae"][indx] = mean_absolute_error(Y_test, preds)
        metrics_["r2"][indx] = r2_score(Y_test, preds)

        indx += 1

    output_mse = generate_score(metrics_["mse"])
    output_mae = generate_score(metrics_["mae"])
    output_r2 = generate_score(metrics_["r2"])

    return {
        "raw": {
            "mse": output_mse,
            "mae": output_mae,
            "r2": output_r2,
        },
        "str": {
            "mse": print_score(output_mse),
            "mae": print_score(output_mae),
            "r2": print_score(output_r2),
        },
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_regression_multiple_seeds(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    Y: Union[pd.Series, np.ndarray, List],
    n_folds: int = 3,
    pretrained: bool = False,
    group_ids: Optional[pd.Series] = None,
    seeds: List[int] = [0, 1, 2],
) -> Dict:
    """Helper for evaluating regression tasks with multiple seeds.

    Args:
        estimator:
            Baseline model to evaluate. if pretrained == False, it must not be fitted.
        X: pd.DataFrame or np.ndarray
            covariates
        Y: pd.Series or np.ndarray or list
            outcomes
        n_folds: int
            Number of cross-validation folds
        seeds: list
            Random seeds
        group_ids: pd.Series
            Optional group_ids for stratified cross-validation

    Returns:
        Dict containing "seeds", "agg" and "str" nodes. The "str" node contains the aggregated prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric. The "seeds" node contains the results for each random seed.
        Both "agg" and "str" nodes contain the following metrics:
            - "r2": R^2(coefficient of determination) regression score function.
            - "mse": Mean squared error regression loss.
            - "mae": Mean absolute error regression loss.
    """
    metrics = reg_supported_metrics

    results = {
        "seeds": {},
        "agg": {},
        "str": {},
    }

    repeats = {}
    for metric in metrics:
        repeats[metric] = []
    for seed in seeds:
        score = evaluate_regression(
            estimator,
            X=X,
            Y=Y,
            n_folds=n_folds,
            metrics=metrics,
            seed=seed,
            pretrained=pretrained,
            group_ids=group_ids,
        )

        results["seeds"][seed] = score["str"]
        for metric in metrics:
            repeats[metric].append(score["raw"][metric][0])

    for metric in metrics:
        output_clf = generate_score(repeats[metric])
        results["agg"][metric] = output_clf
        results["str"][metric] = print_score(output_clf)

    return results


def score_classification_model(
    estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> float:
    model = copy.deepcopy(estimator)
    model.fit(X_train, y_train)

    return model.score(X_test, y_test)


def _constant_columns(dataframe: pd.DataFrame) -> list:
    """
    Drops constant value columns of pandas dataframe.
    """
    result = []
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 1:
            result.append(column)
    return result
