# stdlib
import copy
from typing import Any, Callable, Dict, List, Optional, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import (
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
from autoprognosis.utils.metrics import (
    evaluate_auc,
    evaluate_skurv_brier_score,
    evaluate_skurv_c_index,
    generate_score,
    print_score,
)
from autoprognosis.utils.risk_estimation import generate_dataset_for_horizon

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


class classifier_evaluator:
    """Helper class for evaluating the performance of the classifier.

    Args:
        metric: str, default="aucroc"
            The type of metric to use for evaluation. Potential values: ["aucprc", "aucroc"].
    """

    def __init__(self, metric: str = "aucroc") -> None:
        metric_allowed = ["aucprc", "aucroc"]

        if metric not in metric_allowed:
            raise ValueError(
                f"invalid metric {metric}. supported values are {metric_allowed}"
            )
        self.m_metric = metric

    def get_metric(self) -> str:
        return self.m_metric

    def score_proba(self, y_test: np.ndarray, y_pred_proba: np.ndarray) -> float:
        if y_test is None or y_pred_proba is None:
            raise RuntimeError("Invalid input for score_proba")

        if self.m_metric == "aucprc":
            score_val = self.average_precision_score(y_test, y_pred_proba)
        elif self.m_metric == "aucroc":
            score_val = self.roc_auc_score(y_test, y_pred_proba)
        else:
            raise ValueError(f"invalid metric {self.m_metric}")

        log.debug(f"evaluate:{score_val:0.5f}")
        return score_val

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
    metric: str = "aucroc",
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
        X:
            The covariates
        Y:
            The labels
        n_folds: int
            cross-validation folds
        metric: str
            The metric to use: aucroc or aucprc
        seed: int
            Random seed
        pretrained: bool
            If the estimator was already trained or not.
        group_ids: pd.Series
            The group_ids to use for stratified cross-validation

    """
    X = pd.DataFrame(X).reset_index(drop=True)
    Y = LabelEncoder().fit_transform(Y)
    Y = pd.Series(Y).reset_index(drop=True)
    if group_ids is not None:
        group_ids = pd.Series(group_ids).reset_index(drop=True)

    log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    metric_ = np.zeros(n_folds)

    indx = 0
    if group_ids is not None:
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    ev = classifier_evaluator(metric)

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

        metric_[indx] = ev.score_proba(Y_test, preds)

        indx += 1

    output_clf = generate_score(metric_)

    return {
        "clf": {
            metric: output_clf,
        },
        "str": {
            metric: print_score(output_clf),
        },
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_survival_estimator(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    T: Union[pd.Series, np.ndarray, List],
    Y: Union[pd.Series, np.ndarray, List],
    time_horizons: Union[List[float], np.ndarray],
    n_folds: int = 3,
    metrics: List[str] = survival_supported_metrics,
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
        metrics: list
            Available metrics: "c_index", "brier_score", "aucroc"
        seed: int
            Random seed
        pretrained: bool
            If the estimator was trained or not
        group_ids:
            Group labels for the samples used while splitting the dataset into train/test set.
    """

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

            constant_cols = constant_columns(X_train)
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
                return (
                    fn(
                        T_train,
                        Y_train,
                        pred[:, k],
                        T_test,
                        Y_test,
                        eval_horizon,
                    )
                    / (len(time_horizons))
                )

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

            constant_cols = constant_columns(X_train)
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

    return output


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_regression(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    Y: Union[pd.Series, np.ndarray, List],
    n_folds: int = 3,
    metrics: str = ["rmse", "r2"],
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
        metrics: str
            rmse, r2
        seed: int
            Random seed
        group_ids: pd.Series
            Optional group_ids for stratified cross-validation

    """
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
        },
        "str": {
            "rmse": print_score(output_rmse),
            "r2": print_score(output_r2),
        },
    }


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


def constant_columns(dataframe: pd.DataFrame) -> list:
    """
    Drops constant value columns of pandas dataframe.
    """
    result = []
    for column in dataframe.columns:
        if len(dataframe[column].unique()) == 1:
            result.append(column)
    return result
