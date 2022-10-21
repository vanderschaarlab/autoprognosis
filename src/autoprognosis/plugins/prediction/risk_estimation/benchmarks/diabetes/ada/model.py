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
    fh_diab: int,  # Do immediate family (mother, father, brothers or sisters) have diabetes?
    b_treatedhyp: bool,  # Do you have high blood pressure requiring treatment?
    b_daily_exercise: bool,
    bmi: float,  # Body mass index = kg/m^2
) -> float:
    score = 0.0

    if gender == "M":
        score += 1

    if age >= 40 and age < 50:
        score += 1
    elif age >= 50 and age < 60:
        score += 2
    elif age >= 60:
        score += 3

    if fh_diab:
        score += 1

    if b_treatedhyp:
        score += 1

    if not b_daily_exercise:
        score += 1

    if bmi >= 25 and bmi < 30:
        score += 1
    elif bmi >= 30 and bmi < 40:
        score += 2
    elif bmi >= 40:
        score += 3

    if score > 5:
        return 0.2

    return 0.01


class ADAModel:
    def __init__(self) -> None:
        pass

    def fit(self, *args: Any, **kwargs: Any) -> "ADAModel":
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
                fh_diab=row[
                    "fh_diab"
                ],  # Do immediate family (mother, father, brothers or sisters) have diabetes?
                b_treatedhyp=row["ht_treat"],  # On blood pressure treatment?
                b_daily_exercise=row["b_daily_exercise"],
                bmi=row["bmi"],  # Body mass index = kg/m^2
            )
            return score

        expected_cols = ["sex", "age", "fh_diab", "ht_treat", "b_daily_exercise", "bmi"]
        for col in expected_cols:
            if col not in df.columns:
                log.error(f"[ADA] missing {col}")
                df[col] = 0

        scores = df.apply(lambda row: qdiabetes_inference(row), axis=1)

        return pd.DataFrame(scores, columns=[10 * 365])
