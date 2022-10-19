# 10-year risk CVD prediction

# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.utils.metrics import evaluate_auc

# white women
BETA_WOMEN_W = np.array(
    [
        -29.799,  # natural log age ln age
        4.884,  # ln age squared
        13.540,  # ln total Cholesterol (mg/dL)
        -3.114,  # Ln Age×Ln Total Cholesterol
        -13.578,  # Ln HDL–C
        3.149,  # Ln Age×Ln
        2.019,  # log treated systolic BP (mm Hg)
        0,  # log Age×log treated systolic BP
        1.957,  # log untreated systolic BP
        0,  # log Age×log untreated systolic BP
        7.574,  # smoking (1=yes,0=no)
        -1.665,  # log age× smoking
        0.661,  # diabets
    ]
)

# African American women
BETA_WOMEN_B = np.array(
    [
        17.114,  # natrual log age ln age
        0,  # ln age squared
        0.94,  # ln total Cholesterol (mg/dL)
        0,  # Ln Age×Ln Total Cholesterol
        -18.920,  # Ln HDL–C
        4.475,  # Ln Age×Ln
        29.291,  # log treated systolic BP (mm Hg)
        -6.432,  # log Age×log treated systolic BP
        27.82,  # log untreated systolic BP
        -6.087,  # log Age×log untreated systolic BP
        0.691,  # smoking (1=yes,0=no)
        0,  # log age× smoking
        0.874,  # diabets
    ]
)

# white men
BETA_MEN_W = np.array(
    [
        12.344,  # natrual log age ln age
        11.853,  # ln total Cholesterol (mg/dL)
        -2.664,  # Ln Age×Ln Total Cholesterol
        -7.99,  # Ln HDL–C
        1.769,  # Ln Age×Ln HDL-C
        1.797,  # log treated systolic BP (mm Hg)
        1.764,  # log untreated systolic BP
        0.691,  # smoking (1=yes,0=no)
        0,  # log age× smoking
        0.658,  # diabets
    ]
)

# African American men
BETA_MEN_B = np.array(
    [
        2.469,  # natrual log age ln age
        0.302,  # ln total Cholesterol (mg/dL)
        0,  # Ln Age×Ln Total Cholesterol
        -0.307,  # Ln HDL–C
        0,  # Ln Age×Ln HDL-C
        1.916,  # log treated systolic BP (mm Hg)
        1.809,  # log untreated systolic BP
        0.549,  # smoking (1=yes,0=no)
        0,  # log age× smoking
        0.645,  # diabets
    ]
)
# survival rate baseline
SURV_WOMEN_W = 0.9665
SURV_WOMEN_B = 0.9553
SURV_MEN_W = 0.9144
SURV_MEN_B = 0.8954


def _calc_frs(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.sum(X.dot(beta))


def inference(
    gender: str = "F",
    age: int = 55,
    tchol: float = 213,
    hdlc: float = 50,
    sbp: float = 120,
    smoking: bool = False,
    diab: bool = False,
    ht_treat: bool = False,
    race: str = "W",
) -> float:
    """
    :param gender: 'F' or 'M'
    :param age:
    :param tchol: total cholesterol
    :param hdlc:
    :param sbp: blood pressue
    :param smoking: 0 or 1
    :param diab: 0 or 1 (can be more than 1)
    :param ht_treat:0 or 1
    :param race:
    :return:
    """
    if gender.upper() == "F":
        X_women = np.array(
            [
                np.log(age),
                np.square(np.log(age)),
                np.log(tchol),
                np.log(age) * np.log(tchol),
                np.log(hdlc),
                np.log(age) * np.log(hdlc),
                np.log(sbp) * bool(ht_treat),
                np.log(age) * np.log(sbp) * bool(ht_treat),
                np.log(sbp) * (1 - bool(ht_treat)),
                np.log(age) * np.log(sbp) * (1 - bool(ht_treat)),
                bool(smoking),
                np.log(age) * bool(smoking),
                bool(diab),
            ]
        )
        if race.upper() == "W":
            ind_frs = _calc_frs(X_women, BETA_WOMEN_W)
            mean_frs = -29.18
            score = 1 - np.power(SURV_WOMEN_W, np.exp(ind_frs - mean_frs))
        elif race.upper() == "B":
            ind_frs = _calc_frs(X_women, BETA_WOMEN_B)
            mean_frs = 86.61
            score = 1 - np.power(SURV_WOMEN_B, np.exp(ind_frs - mean_frs))
        else:
            raise ValueError("Race must be specified as W or B")
    elif gender.upper() == "M":

        X_men = np.array(
            [
                np.log(age),
                np.log(tchol),
                np.log(age) * np.log(tchol),
                np.log(hdlc),
                np.log(age) * np.log(hdlc),
                np.log(sbp) * bool(ht_treat),
                np.log(sbp) * (1 - bool(ht_treat)),
                bool(smoking),
                np.log(age) * bool(smoking),
                bool(diab),
            ]
        )
        if race.upper() == "W":
            ind_frs = _calc_frs(X_men, BETA_MEN_W)
            mean_frs = 61.18
            score = 1 - np.power(SURV_MEN_W, np.exp(ind_frs - mean_frs))
        elif race.upper() == "B":
            ind_frs = _calc_frs(X_men, BETA_MEN_B)
            mean_frs = 19.54
            score = 1 - np.power(SURV_MEN_B, np.exp(ind_frs - mean_frs))
        else:
            raise ValueError("Race must be specified as W or B")
    else:
        raise ValueError("Gender must be specified as M or F")

    return score


def mmolL_to_mgdl(val: float) -> float:
    return val * 18.0182


class AHAModel:
    def __init__(self) -> None:
        pass

    def fit(self, *args: Any, **kwargs: Any) -> "AHAModel":
        return self

    def score(self, X: pd.DataFrame, Y: pd.DataFrame) -> float:
        local_preds = self.predict(X).to_numpy()[:, 0]
        local_surv_pred = 1 - local_preds

        full_proba = []
        full_proba.append(local_surv_pred)
        full_proba.append(local_preds)
        full_proba = pd.DataFrame(full_proba).T

        return evaluate_auc(Y, full_proba)[0]

    def predict(
        self, df: pd.DataFrame, times: list = []
    ) -> pd.DataFrame:  # times is considered always ten years
        def aha_inference(row: pd.DataFrame) -> pd.DataFrame:
            tchol = mmolL_to_mgdl(row["tchol"])
            hdlc = mmolL_to_mgdl(row["hdl"])

            score = inference(
                gender=row["sex"],
                age=row["age"],
                tchol=tchol,
                hdlc=hdlc,
                sbp=row["sbp"],
                smoking=row["smoker"],
                diab=row["diabetes"],
                ht_treat=row["ht_treat"],
                race=row["race"],
            )
            return score

        scores = df.apply(lambda row: aha_inference(row), axis=1)

        return pd.DataFrame(scores, columns=[10 * 365])
