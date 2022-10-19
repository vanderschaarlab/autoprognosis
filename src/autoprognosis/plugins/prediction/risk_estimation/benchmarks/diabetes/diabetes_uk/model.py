# stdlib
from typing import Any

# third party
import pandas as pd

# autoprognosis absolute
from autoprognosis.utils.metrics import evaluate_auc


def inference(
    gender: str,  # M/F
    age: float,  # age value
    ethrisk: int,  # ethnic risk
    fh_diab: int,  # Do immediate family (mother, father, brothers or sisters) have diabetes?
    waist: float,  # waist size
    bmi: float,  # Body mass index = kg/m^2
    b_treatedhyp: bool,  # Do you have high blood pressure requiring treatment?
) -> float:

    score = 0.0
    if age >= 50 and age <= 59:
        score += 5
    elif age >= 60 and age <= 69:
        score += 9
    elif age >= 70:
        score += 13

    if gender == "M":
        score += 1

    if ethrisk != 0:
        score += 6

    if fh_diab > 0:
        score += 5

    if waist >= 90 and waist < 100:
        score += 4
    elif waist >= 100 and waist < 110:
        score += 6
    elif waist >= 110:
        score += 9

    if bmi >= 25 and bmi < 30:
        score += 3
    elif bmi >= 30 and bmi < 35:
        score += 5
    elif bmi >= 35:
        score += 8

    if b_treatedhyp:
        score += 5

    if score <= 6:
        return 1 / 20
    elif score <= 15:
        return 1 / 10
    elif score <= 24:
        return 1 / 7
    else:
        return 1 / 3


class DiabetesUKModel:
    def __init__(self) -> None:
        pass

    def fit(self, *args: Any, **kwargs: Any) -> "DiabetesUKModel":
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
                ethrisk=row["ethrisk"],  # ethnic risk
                fh_diab=row[
                    "fh_diab"
                ],  # Do immediate family (mother, father, brothers or sisters) have diabetes?
                waist=row["waist"],
                bmi=row["bmi"],  # Body mass index = kg/m^2
                b_treatedhyp=row["ht_treat"],  # On blood pressure treatment?
            )
            return score

        scores = df.apply(lambda row: qdiabetes_inference(row), axis=1)

        return pd.DataFrame(scores, columns=[10 * 365])
