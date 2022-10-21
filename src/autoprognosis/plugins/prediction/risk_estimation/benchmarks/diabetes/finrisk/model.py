# stdlib
from typing import Any

# third party
import pandas as pd

# autoprognosis absolute
import autoprognosis.logger as log
from autoprognosis.utils.metrics import evaluate_auc


def inference(
    gender: str,  # M/F
    age: float,  # age value
    bmi: float,  # Body mass index = kg/m^2
    waist: float,  # waist size
    b_daily_exercise: bool,
    b_daily_vegs: bool,
    b_treatedhyp: bool,  # Do you have high blood pressure requiring treatment?
    b_ever_had_high_glucose: bool,
    fh_diab: int,  # Do immediate family (mother, father, brothers or sisters) have diabetes?
) -> float:

    score = 0.0

    if age >= 45 and age < 55:
        score += 2
    elif age >= 55 and age < 65:
        score += 3
    elif age >= 65:
        score += 4

    if bmi >= 25 and bmi <= 30:
        score += 1
    elif bmi > 30:
        score += 3

    if gender == "M":
        if waist >= 94 and waist <= 102:
            score += 3
        elif waist > 102:
            score += 4
    else:
        if waist >= 80 and waist <= 88:
            score += 3
        elif waist > 88:
            score += 4

    if not b_daily_exercise:
        score += 2

    if not b_daily_vegs:
        score += 1

    if b_treatedhyp:
        score += 2

    if b_ever_had_high_glucose:
        score += 5

    if fh_diab:
        score += 5

    if score < 7:
        return 1 / 100
    elif score <= 11:
        return 1 / 25
    elif score <= 14:
        return 1 / 6
    elif score <= 20:
        return 1 / 3
    else:
        return 1 / 2


class FINRISKModel:
    def __init__(self) -> None:
        pass

    def fit(self, *args: Any, **kwargs: Any) -> "FINRISKModel":
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
    ) -> Any:  # times is considered always ten years
        def qdiabetes_inference(row: pd.DataFrame) -> Any:
            score = inference(
                gender=row["sex"],  # M/F
                age=row["age"],  # age value
                bmi=row["bmi"],  # Body mass index = kg/m^2
                waist=row["waist"],
                b_daily_exercise=row["b_daily_exercise"],
                b_daily_vegs=row["b_daily_vegs"],
                b_treatedhyp=row["ht_treat"],  # On blood pressure treatment?
                b_ever_had_high_glucose=row["b_ever_had_high_glucose"],
                fh_diab=row[
                    "fh_diab"
                ],  # Do immediate family (mother, father, brothers or sisters) have diabetes?
            )
            return score

        expected_cols = [
            "sex",
            "age",
            "fh_diab",
            "ht_treat",
            "b_daily_exercise",
            "bmi",
            "waist",
            "b_daily_vegs",
            "b_ever_had_high_glucose",
        ]
        for col in expected_cols:
            if col not in df.columns:
                log.error(f"[ADA] missing {col}")
                df[col] = 0

        scores = df.apply(lambda row: qdiabetes_inference(row), axis=1)

        return pd.DataFrame(scores, columns=[10 * 365])
