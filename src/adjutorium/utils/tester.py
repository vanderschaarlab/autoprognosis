# stdlib
import copy
from typing import Any, Callable, Dict, List

# third party
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

# adjutorium absolute
import adjutorium.logger as log
from adjutorium.utils.metrics import (
    evaluate_auc,
    evaluate_skurv_brier_score,
    evaluate_skurv_c_index,
    generate_score,
    print_score,
)
from adjutorium.utils.risk_estimation import generate_dataset_for_horizon


class Eval:
    """Helper class for evaluating the performance of the models.

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
        assert y_test is not None
        assert y_pred_proba is not None

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


def evaluate_estimator(
    estimator: Any,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_folds: int = 3,
    metric: str = "aucroc",
    seed: int = 0,
    pretrained: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Dict:
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    metric_ = np.zeros(n_folds)

    indx = 0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    ev = Eval(metric)

    for train_index, test_index in skf.split(X, Y):

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


def evaluate_survival_estimator(
    estimator: Any,
    X: pd.DataFrame,
    T: pd.DataFrame,
    Y: pd.DataFrame,
    time_horizons: List,
    n_folds: int = 3,
    metrics: List[str] = ["c_index", "brier_score", "aucroc"],
    seed: int = 0,
    pretrained: bool = False,
) -> Dict:

    supported_metrics = ["c_index", "brier_score", "aucroc"]
    results = {}

    for metric in metrics:
        if metric not in supported_metrics:
            raise ValueError(f"Metric {metric} not supported")

        results[metric] = np.zeros(n_folds)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    cv_idx = 0
    for train_index, test_index in skf.split(X, Y):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        T_train = T.loc[T.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]
        T_test = T.loc[T.index[test_index]]

        if pretrained:
            model = estimator[cv_idx]
        else:
            model = copy.deepcopy(estimator)
            model.fit(X_train, T_train, Y_train)

        try:
            pred = model.predict(X_test, time_horizons).to_numpy()
        except BaseException as e:
            raise e

        for k in range(len(time_horizons)):

            def get_score(fn: Callable) -> float:
                return (
                    fn(
                        T_train,
                        Y_train,
                        pred[:, k],
                        T_test,
                        Y_test,
                        time_horizons[k],
                    )
                    / (len(time_horizons))
                )

            for metric in metrics:
                if metric == "c_index":
                    results[metric][cv_idx] += get_score(evaluate_skurv_c_index)
                elif metric == "brier_score":
                    results[metric][cv_idx] += get_score(evaluate_skurv_brier_score)

        cv_idx += 1

    for k in range(len(time_horizons)):
        cv_idx = 0

        X_horizon, T_horizon, Y_horizon = generate_dataset_for_horizon(
            X, T, Y, time_horizons[k]
        )
        for train_index, test_index in skf.split(X_horizon, Y_horizon):

            X_train = X_horizon.loc[X_horizon.index[train_index]]
            Y_train = Y_horizon.loc[Y_horizon.index[train_index]]
            T_train = T_horizon.loc[T_horizon.index[train_index]]
            X_test = X_horizon.loc[X_horizon.index[test_index]]
            Y_test = Y_horizon.loc[Y_horizon.index[test_index]]
            T_test = T_horizon.loc[T_horizon.index[test_index]]

            if pretrained:
                model = estimator[cv_idx]
            else:
                model = copy.deepcopy(estimator)
                model.fit(X_train, T_train, Y_train)

            try:
                pred = model.predict(X_test, time_horizons).to_numpy()
            except BaseException as e:
                raise e

            metric = "aucroc"

            local_preds = pd.DataFrame(pred[:, k]).squeeze()
            local_surv_pred = 1 - local_preds

            full_proba = []
            full_proba.append(local_surv_pred.values)
            full_proba.append(local_preds.values)
            full_proba = pd.DataFrame(full_proba).T

            results[metric][cv_idx] += evaluate_auc(Y_test, full_proba)[0] / (
                len(time_horizons)
            )

            cv_idx += 1

    output: dict = {
        "clf": {},
        "str": {},
    }

    for metric in metrics:
        output["clf"][metric] = generate_score(results[metric])
        output["str"][metric] = print_score(output["clf"][metric])

    return output


def evaluate_survival_classifier(
    estimator: Any,
    X: pd.DataFrame,
    T: pd.DataFrame,
    Y: pd.DataFrame,
    eval_time: float,
    n_folds: int = 3,
    seed: int = 0,
    pretrained: bool = False,
) -> dict:

    metric_c_index = np.zeros(n_folds)
    metric_brier_score = np.zeros(n_folds)
    metric_aucroc = np.zeros(n_folds)

    indx = 0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in skf.split(X, Y):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        T_train = T.loc[T.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]
        T_test = T.loc[T.index[test_index]]

        if pretrained:
            model = estimator[indx]
        else:
            model = copy.deepcopy(estimator)
            model.fit(X_train, Y_train)

        preds = model.predict_proba(X_test)
        preds = np.asarray(preds)

        metric_aucroc[indx] = evaluate_auc(Y_test, preds)[0]

        metric_c_index[indx] = evaluate_skurv_c_index(
            T_train, Y_train, preds[:, 1], T_test, Y_test, eval_time
        )

        metric_brier_score[indx] = evaluate_skurv_brier_score(
            T_train, Y_train, preds[:, 1], T_test, Y_test, eval_time
        )
        indx += 1

    output_cindex = generate_score(metric_c_index)
    output_brier = generate_score(metric_brier_score)
    output_roc = generate_score(metric_aucroc)

    return {
        "clf": {
            "c_index": output_cindex,
            "brier_score": output_brier,
            "aucroc": output_roc,
        },
        "str": {
            "c_index": print_score(output_cindex),
            "brier_score": print_score(output_brier),
            "aucroc": print_score(output_roc),
        },
    }


def evaluate_treatments_model(
    estimator: Any,
    X: pd.DataFrame,
    W: pd.DataFrame,
    Y: pd.DataFrame,
    Y_full: pd.DataFrame,
    n_folds: int = 3,
    seed: int = 0,
    pretrained: bool = False,
) -> dict:
    X = np.asarray(X)
    W = np.asarray(W)
    Y = np.asarray(Y)
    Y_full = np.asarray(Y_full)

    metric_pehe = np.zeros(n_folds)
    metric_ate = np.zeros(n_folds)

    indx = 0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in skf.split(X, Y):

        X_train = X[train_index]
        Y_train = Y[train_index]
        W_train = W[train_index]

        X_test = X[test_index]
        Y_full_test = Y_full[test_index]

        if pretrained:
            model = estimator[indx]
        else:
            model = copy.deepcopy(estimator)
            model.fit(X_train, W_train, Y_train)

        metric_pehe[indx] = model.score(X_test, Y_full_test, metric="pehe")
        metric_ate[indx] = model.score(X_test, Y_full_test, metric="ate")
        indx += 1

    output_pehe = generate_score(metric_pehe)
    output_ate = generate_score(metric_ate)

    return {
        "clf": {
            "pehe": output_pehe,
            "ate": output_ate,
        },
        "str": {
            "pehe": print_score(output_pehe),
            "ate": print_score(output_ate),
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


def score_treatments_model(
    estimator: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    T_train: pd.DataFrame,
) -> float:
    model = copy.deepcopy(estimator)
    model.fit(X_train, T_train, y_train)

    return model.score(X_test, y_test)


def evaluate_regression(
    estimator: Any,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    n_folds: int = 3,
    metric: str = "rmse",
    seed: int = 0,
    *args: Any,
    **kwargs: Any,
) -> Dict:
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)

    log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    metric_ = np.zeros(n_folds)

    indx = 0
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in skf.split(X, Y):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        model = copy.deepcopy(estimator)
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        metric_[indx] = mean_squared_error(Y_test, preds)

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
