# stdlib
from typing import Tuple, Union

# third party
from lifelines import KaplanMeierFitter
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# adjutorium absolute
import adjutorium.logger as log
from adjutorium.utils.third_party.metrics import brier_score, concordance_index_ipcw


def get_y_pred_proba_hlpr(y_pred_proba: np.ndarray, nclasses: int) -> np.ndarray:
    if nclasses == 2:
        if len(y_pred_proba.shape) < 2:
            return y_pred_proba

        if y_pred_proba.shape[1] == 2:
            return y_pred_proba[:, 1]

    return y_pred_proba


def evaluate_auc(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    classes: Union[np.ndarray, None] = None,
) -> Tuple[float, float]:

    y_test = np.asarray(y_test)
    y_pred_proba = np.asarray(y_pred_proba)

    nnan = sum(np.ravel(np.isnan(y_pred_proba)))

    if nnan:
        raise ValueError("nan in predictions. aborting")

    n_classes = len(set(np.ravel(y_test)))

    y_pred_proba_tmp = get_y_pred_proba_hlpr(y_pred_proba, n_classes)

    if n_classes > 2:

        log.debug(f"+evaluate_auc {y_test.shape} {y_pred_proba_tmp.shape}")

        fpr = dict()
        tpr = dict()
        precision = dict()
        recall = dict()
        average_precision = dict()
        roc_auc: dict = dict()

        if classes is None:
            classes = sorted(set(np.ravel(y_test)))
            log.debug(
                "warning: classes is none and more than two "
                " (#{}), classes assumed to be an ordered set:{}".format(
                    n_classes, classes
                )
            )

        y_test = label_binarize(y_test, classes=classes)

        fpr["micro"], tpr["micro"], _ = roc_curve(
            y_test.ravel(), y_pred_proba_tmp.ravel()
        )
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_test.ravel(), y_pred_proba_tmp.ravel()
        )

        average_precision["micro"] = average_precision_score(
            y_test, y_pred_proba_tmp, average="micro"
        )

        aucroc = roc_auc["micro"]
        aucprc = average_precision["micro"]
    else:

        aucroc = roc_auc_score(np.ravel(y_test), y_pred_proba_tmp)
        aucprc = average_precision_score(np.ravel(y_test), y_pred_proba_tmp)

    return aucroc, aucprc


def censoring_probability(Y: np.ndarray, T: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y)
    T = np.asarray(T)

    T = T.reshape([-1])  # (N,) - np array
    Y = Y.reshape([-1])  # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(
        T, event_observed=(Y == 0).astype(int)
    )  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][
        -1
    ]  # fill 0 with ZoH (to prevent nan values)

    return G


def evaluate_weighted_brier_score(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float,
) -> np.ndarray:
    censoring_probs = censoring_probability(Y_train, T_train)
    Prediction = np.asarray(Prediction)
    T_test = np.asarray(T_test)
    Y_test = np.asarray(Y_test)

    G = censoring_probs
    N = len(Prediction)

    W = np.zeros(len(Y_test))
    Y_tilde = (T_test > Time).astype(float)

    for i in range(N):
        tmp_idx1 = np.where(G[0, :] >= T_test[i])[0]
        tmp_idx2 = np.where(G[0, :] >= Time)[0]

        if len(tmp_idx1) == 0:
            G1 = G[1, -1]
        else:
            G1 = G[1, tmp_idx1[0]]

        if len(tmp_idx2) == 0:
            G2 = G[1, -1]
        else:
            G2 = G[1, tmp_idx2[0]]
        W[i] = (1.0 - Y_tilde[i]) * float(Y_test[i]) / G1 + Y_tilde[i] / G2

    return np.mean(W * (Y_tilde - (1.0 - Prediction)) ** 2)


def evaluate_weighted_c_index(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float,
) -> float:
    censoring_probs = censoring_probability(Y_train, T_train)
    Prediction = np.asarray(Prediction)
    T_test = np.asarray(T_test)
    Y_test = np.asarray(Y_test)

    G = censoring_probs

    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0
    for i in range(N):
        tmp_idx = np.where(G[0, :] >= T_test[i])[0]

        if len(tmp_idx) == 0:
            W = (1.0 / G[1, -1]) ** 2
        else:
            W = (1.0 / G[1, tmp_idx[0]]) ** 2

        A[i, np.where(T_test[i] < T_test)] = 1.0 * W
        Q[i, np.where(Prediction[i] > Prediction)] = 1.0  # give weights

        if T_test[i] <= Time and Y_test[i] == 1:
            N_t[i, :] = 1.0

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    if Num == 0 and Den == 0:
        result = float(-1)  # not able to compute c-index!
    else:
        result = float(Num / Den)

    return result


def evaluate_skurv_c_index(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float,
) -> float:
    T_train = pd.Series(T_train)
    Y_train = pd.Series(Y_train)
    T_test = pd.Series(T_test)
    Y_test = pd.Series(Y_test)

    Y_train_structured = [
        (Y_train.iloc[i], T_train.iloc[i]) for i in range(len(Y_train))
    ]
    Y_train_structured = np.array(
        Y_train_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    Y_test_structured = [(Y_test.iloc[i], T_test.iloc[i]) for i in range(len(Y_test))]
    Y_test_structured = np.array(
        Y_test_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    try:
        # concordance_index_ipcw expects risk scores
        return concordance_index_ipcw(
            Y_train_structured, Y_test_structured, Prediction, tau=Time
        )
    except BaseException:
        return evaluate_weighted_c_index(
            T_train, Y_train, Prediction, T_test, Y_test, Time
        )


def evaluate_skurv_brier_score(
    T_train: np.ndarray,
    Y_train: np.ndarray,
    Prediction: np.ndarray,
    T_test: np.ndarray,
    Y_test: np.ndarray,
    Time: float,
) -> float:
    T_train = pd.Series(T_train)
    Y_train = pd.Series(Y_train)
    T_test = pd.Series(T_test)
    Y_test = pd.Series(Y_test)

    Y_train_structured = [
        (Y_train.iloc[i], T_train.iloc[i]) for i in range(len(Y_train))
    ]
    Y_train_structured = np.array(
        Y_train_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    Y_test_structured = [(Y_test.iloc[i], T_test.iloc[i]) for i in range(len(Y_test))]
    Y_test_structured = np.array(
        Y_test_structured, dtype=[("status", "bool"), ("time", "<f8")]
    )

    # brier_score expects survival scores
    try:
        return brier_score(
            Y_train_structured, Y_test_structured, 1 - Prediction, times=Time
        )[0]
    except BaseException:
        return evaluate_weighted_brier_score(
            T_train, Y_train, Prediction, T_test, Y_test, Time
        )


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 4)) + " +/- " + str(round(score[1], 4))
