# stdlib
import copy
from typing import Any, Callable, Dict, List, Optional, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
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
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
import autoprognosis.logger as log
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.metrics import (
    evaluate_auc,
    evaluate_skurv_brier_score,
    evaluate_skurv_c_index,
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
reg_supported_metrics = ["rmse", "r2"]


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
                - "kappa":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
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
                results[metric] = f1_score(y_test, y_pred, average="micro")
            elif metric == "f1_score_macro":
                results[metric] = f1_score(y_test, y_pred, average="macro")
            elif metric == "f1_score_weighted":
                results[metric] = f1_score(y_test, y_pred, average="weighted")
            elif metric == "kappa":
                results[metric] = cohen_kappa_score(y_test, y_pred)
            elif metric == "recall_micro":
                results[metric] = recall_score(y_test, y_pred, average="micro")
            elif metric == "recall_macro":
                results[metric] = recall_score(y_test, y_pred, average="macro")
            elif metric == "recall_weighted":
                results[metric] = recall_score(y_test, y_pred, average="weighted")
            elif metric == "precision_micro":
                results[metric] = precision_score(y_test, y_pred, average="micro")
            elif metric == "precision_macro":
                results[metric] = precision_score(y_test, y_pred, average="macro")
            elif metric == "precision_weighted":
                results[metric] = precision_score(y_test, y_pred, average="weighted")
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
        "clf": output_clf,  # legacy
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
        for metric in score["clf"]:
            repeats[metric].append(score["clf"][metric][0])

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
        X: DataFrame
            The covariates
        T: Series
            time to event
        Y: Series
            event or censored
        time_horizons: list
            Horizons where to evaluate the performance.
        n_folds: int
            Number of folds for cross validation
        seed: int
            Random seed
        pretrained: bool
            If the estimator was trained or not
        group_ids:
            Group labels for the samples used while splitting the dataset into train/test set.
    """
    enable_reproducible_results(seed)
    metrics = survival_supported_metrics

    results = {}
    X = pd.DataFrame(X).reset_index(drop=True)
    Y = pd.Series(Y).reset_index(drop=True)
    T = pd.Series(T).reset_index(drop=True)
    if group_ids is not None:
        group_ids = pd.Series(group_ids).reset_index(drop=True)

    for metric in metrics:
        if metric not in survival_supported_metrics:
            raise ValueError(f"Metric {metric} not supported")

        results[metric] = np.zeros(n_folds)

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

        try:
            pred = model.predict(X_test, time_horizons).to_numpy()
        except BaseException as e:
            raise e

        c_index = 0.0
        brier_score = 0.0

        for k in range(len(time_horizons)):
            eval_horizon = min(time_horizons[k], np.max(T_test) - 1)

            def get_score(fn: Callable) -> float:
                return fn(
                    T_train,
                    Y_train,
                    pred[:, k],
                    T_test,
                    Y_test,
                    eval_horizon,
                ) / (len(time_horizons))

            c_index += get_score(evaluate_skurv_c_index)
            brier_score += get_score(evaluate_skurv_brier_score)

        return c_index, brier_score

    def _get_clf_metrics(
        cv_idx: int,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        T_train: pd.DataFrame,
        T_test: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
        time_horizons: list,
    ) -> Dict[str, float]:
        cv_idx = 0

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

        try:
            pred = model.predict(X_test, time_horizons).to_numpy()
        except BaseException as e:
            raise e

        local_scores = pd.DataFrame(pred[:, k]).squeeze()
        local_preds = (local_scores > risk_threshold).astype(int)

        return {
            "aucroc": roc_auc_score(Y_test, local_scores) / (len(time_horizons)),
            "specificity": recall_score(Y_test, local_preds, pos_label=0)
            / (len(time_horizons)),
            "sensitivity": recall_score(Y_test, local_preds, pos_label=1)
            / (len(time_horizons)),
            "PPV": precision_score(Y_test, local_preds, pos_label=1)
            / (len(time_horizons)),
            "NPV": precision_score(Y_test, local_preds, pos_label=0)
            / (len(time_horizons)),
            "predicted_cases": local_preds.sum(),
        }

    if n_folds == 1:
        cv_idx = 0
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y)
        local_time_horizons = [t for t in time_horizons if t > np.min(T_test)]

        c_index, brier_score = _get_surv_metrics(
            cv_idx,
            X_train,
            X_test,
            T_train,
            T_test,
            Y_train,
            Y_test,
            local_time_horizons,
        )
        for metric in metrics:
            if metric == "c_index":
                results[metric][cv_idx] = c_index
            elif metric == "brier_score":
                results[metric][cv_idx] = brier_score

        for k in range(len(time_horizons)):
            cv_idx = 0

            X_horizon, T_horizon, Y_horizon = generate_dataset_for_horizon(
                X, T, Y, time_horizons[k]
            )
            X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
                X_horizon, T_horizon, Y_horizon
            )

            clf_metrics = _get_clf_metrics(
                cv_idx,
                X_train,
                X_test,
                T_train,
                T_test,
                Y_train,
                Y_test,
                local_time_horizons,
            )
            for metric in clf_metrics:
                if metric in metrics:
                    results[metric][cv_idx] += clf_metrics[metric]

    else:
        if group_ids is not None:
            skf = StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=seed
            )
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

            c_index, brier_score = _get_surv_metrics(
                cv_idx,
                X_train,
                X_test,
                T_train,
                T_test,
                Y_train,
                Y_test,
                local_time_horizons,
            )
            for metric in metrics:
                if metric == "c_index":
                    results[metric][cv_idx] = c_index
                elif metric == "brier_score":
                    results[metric][cv_idx] = brier_score

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
                    local_time_horizons,
                )
                for metric in clf_metrics:
                    if metric in metrics:
                        results[metric][cv_idx] += clf_metrics[metric]

                cv_idx += 1

    output: dict = {
        "clf": {},
        "str": {},
    }

    for metric in metrics:
        output["clf"][metric] = generate_score(results[metric])
        output["str"][metric] = print_score(output["clf"][metric])

    output["raw"] = output["clf"]

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
        X: DataFrame
            The covariates
        T: Series
            time to event
        Y: Series
            event or censored
        time_horizons: list
            Horizons where to evaluate the performance.
        n_folds: int
            Number of folds for cross validation
        seeds: List
            Random seeds
        pretrained: bool
            If the estimator was trained or not
        group_ids:
            Group labels for the samples used while splitting the dataset into train/test set.
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
            repeats[metric].append(score["clf"][metric][0])

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
            The regressor to evaluate
        X:
            covariates
        Y:
            outcomes
        n_folds: int
            Number of cross-validation folds
        seed: int
            Random seed
        group_ids: pd.Series
            Optional group_ids for stratified cross-validation

    """
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

        metrics_["rmse"][indx] = mean_squared_error(Y_test, preds)
        metrics_["r2"][indx] = r2_score(Y_test, preds)

        indx += 1

    output_rmse = generate_score(metrics_["rmse"])
    output_r2 = generate_score(metrics_["r2"])

    return {
        "clf": {
            "rmse": output_rmse,
            "r2": output_r2,
        },  # legacy node
        "raw": {
            "rmse": output_rmse,
            "r2": output_r2,
        },
        "str": {
            "rmse": print_score(output_rmse),
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
            The regressor to evaluate
        X:
            covariates
        Y:
            outcomes
        n_folds: int
            Number of cross-validation folds
        seeds: list
            Random seeds
        group_ids: pd.Series
            Optional group_ids for stratified cross-validation

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
            repeats[metric].append(score["clf"][metric][0])

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
