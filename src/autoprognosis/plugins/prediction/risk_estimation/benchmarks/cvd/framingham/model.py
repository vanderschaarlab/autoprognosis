# stdlib
from typing import Any

# third party
import pandas as pd

# autoprognosis absolute
from autoprognosis.utils.metrics import evaluate_auc


def inference(
    sex: str,
    age: int,
    total_cholesterol: float,  # mg/dL
    hdl_cholesterol: float,  # mg/dL
    systolic_blood_pressure: int,
    smoker: bool,
    blood_pressure_med_treatment: bool,
) -> float:
    """Requires:
    sex                             - "M" or "F" string
    age                             - int
    total_cholesterol               - int
    hdl_cholesterol                 - int
    systolic_blood_pressure         - int
    smoker                          - True or False.
    blood_pressure_med_treatment    - True or False.
    """

    # intialize some things -----------------------------------------------------
    points = 0

    # Process males -----------------------------------------------------------
    if sex.lower() == "m":
        # Age - male
        if age <= 34:
            points -= 9
        if 35 <= age <= 39:
            points -= 4
        if 40 <= age <= 44:
            points -= 0
        if 45 <= age <= 49:
            points += 3
        if 50 <= age <= 54:
            points += 6
        if 55 <= age <= 59:
            points += 8
        if 60 <= age <= 64:
            points += 10
        if 65 <= age <= 69:
            points += 12
        if 70 <= age <= 74:
            points += 14
        if 75 <= age:
            points += 16

        # Total cholesterol, mg/dL - Male ------------------------
        if age <= 39:
            if total_cholesterol < 160:
                points += 0
            if 160 <= total_cholesterol <= 199:
                points += 4
            if 200 <= total_cholesterol <= 239:
                points += 7
            if 240 <= total_cholesterol <= 279:
                points += 9
            if total_cholesterol > 289:
                points += 11
        if 40 <= age <= 49:
            if total_cholesterol < 160:
                points += 0
            if 160 <= total_cholesterol <= 199:
                points += 3
            if 200 <= total_cholesterol <= 239:
                points += 5
            if 240 <= total_cholesterol <= 279:
                points += 6
            if total_cholesterol > 289:
                points += 8
        if 50 <= age <= 59:
            if total_cholesterol < 160:
                points += 0
            if 160 <= total_cholesterol <= 199:
                points += 2
            if 200 <= total_cholesterol <= 239:
                points += 3
            if 240 <= total_cholesterol <= 279:
                points += 4
            if total_cholesterol > 289:
                points += 5
        if 60 <= age <= 69:
            if total_cholesterol < 160:
                points += 0
            if 160 <= total_cholesterol <= 199:
                points += 1
            if 200 <= total_cholesterol <= 239:
                points += 1
            if 240 <= total_cholesterol <= 279:
                points += 2
            if total_cholesterol > 289:
                points += 3
        if 70 <= age:
            if total_cholesterol < 160:
                points += 0
            if 160 <= total_cholesterol <= 199:
                points += 0
            if 200 <= total_cholesterol <= 239:
                points += 0
            if 240 <= total_cholesterol <= 279:
                points += 1
            if total_cholesterol > 289:
                points += 1
        # smoking - male
        if smoker:
            if age <= 39:
                points += 8
            if 40 <= age <= 49:
                points += 5
            if 50 <= age <= 59:
                points += 3
            if 60 <= age <= 69:
                points += 1
            if 70 <= age:
                points += 1
        else:  # nonsmoker
            points += 0

        # hdl cholesterol
        if hdl_cholesterol > 60:
            points -= 1
        if 50 <= hdl_cholesterol <= 59:
            points += 0
        if 40 <= hdl_cholesterol <= 49:
            points += 1
        if hdl_cholesterol < 40:
            points += 2

        # systolic blood pressure
        if not blood_pressure_med_treatment:
            if systolic_blood_pressure < 120:
                points += 0
            if 120 <= systolic_blood_pressure <= 129:
                points += 0
            if 130 <= systolic_blood_pressure <= 139:
                points += 1
            if 140 <= systolic_blood_pressure <= 159:
                points += 1
            if systolic_blood_pressure >= 160:
                points += 2
        else:  # if the patient is on blood pressure meds
            if systolic_blood_pressure < 120:
                points += 0
            if 120 <= systolic_blood_pressure <= 129:
                points += 1
            if 130 <= systolic_blood_pressure <= 139:
                points += 1
            if 140 <= systolic_blood_pressure <= 159:
                points += 2
            if systolic_blood_pressure >= 160:
                points += 3

        # calulate % risk for males
        if points <= 0:
            percent_risk = 0.1
        elif points == 1:
            percent_risk = 1
        elif points == 2:
            percent_risk = 1
        elif points == 3:
            percent_risk = 1
        elif points == 4:
            percent_risk = 1
        elif points == 5:
            percent_risk = 2
        elif points == 6:
            percent_risk = 2
        elif points == 7:
            percent_risk = 2
        elif points == 8:
            percent_risk = 2
        elif points == 9:
            percent_risk = 5
        elif points == 10:
            percent_risk = 6
        elif points == 11:
            percent_risk = 8
        elif points == 12:
            percent_risk = 10
        elif points == 13:
            percent_risk = 12
        elif points == 14:
            percent_risk = 16
        elif points == 15:
            percent_risk = 20
        elif points == 16:
            percent_risk = 25
        elif points >= 17:
            percent_risk = 30

    # process females ----------------------------------------------------------
    else:
        # Age - female
        if age <= 34:
            points -= 7
        if 35 <= age <= 39:
            points -= 3
        if 40 <= age <= 44:
            points -= 0
        if 45 <= age <= 49:
            points += 3
        if 50 <= age <= 54:
            points += 6
        if 55 <= age <= 59:
            points += 8
        if 60 <= age <= 64:
            points += 10
        if 65 <= age <= 69:
            points += 12
        if 70 <= age <= 74:
            points += 14
        if 75 <= age:
            points += 16

        # Total cholesterol, mg/dL - Female ------------------------
        if age <= 39:
            if total_cholesterol < 160:
                points += 0
            if 160 <= total_cholesterol <= 199:
                points += 4
            if 200 <= total_cholesterol <= 239:
                points += 8
            if 240 <= total_cholesterol <= 279:
                points += 11
            if total_cholesterol > 289:
                points += 13
        if 40 <= age <= 49:
            if total_cholesterol < 160:
                points += 0
            if 160 <= total_cholesterol <= 199:
                points += 3
            if 200 <= total_cholesterol <= 239:
                points += 6
            if 240 <= total_cholesterol <= 279:
                points += 8
            if total_cholesterol > 289:
                points += 10
        if 50 <= age <= 59:
            if total_cholesterol < 160:
                points += 0
            if 160 <= total_cholesterol <= 199:
                points += 2
            if 200 <= total_cholesterol <= 239:
                points += 4
            if 240 <= total_cholesterol <= 279:
                points += 5
            if total_cholesterol > 289:
                points += 7
        if 60 <= age <= 69:
            if total_cholesterol < 160:
                points += 0
            if 160 <= total_cholesterol <= 199:
                points += 1
            if 200 <= total_cholesterol <= 239:
                points += 2
            if 240 <= total_cholesterol <= 279:
                points += 3
            if total_cholesterol > 289:
                points += 4
        if 70 <= age:
            if total_cholesterol < 160:
                points += 0
            if 160 <= total_cholesterol <= 199:
                points += 1
            if 200 <= total_cholesterol <= 239:
                points += 1
            if 240 <= total_cholesterol <= 279:
                points += 2
            if total_cholesterol > 289:
                points += 2
        # smoking - female
        if smoker:
            if age <= 39:
                points += 9
            if 40 <= age <= 49:
                points += 7
            if 50 <= age <= 59:
                points += 4
            if 60 <= age <= 69:
                points += 2
            if 70 <= age:
                points += 1
        else:  # nonsmoker
            points += 0

        # hdl cholesterol - female
        if hdl_cholesterol > 60:
            points -= 1
        if 50 <= hdl_cholesterol <= 59:
            points += 0
        if 40 <= hdl_cholesterol <= 49:
            points += 1
        if hdl_cholesterol < 40:
            points += 2

        # systolic blood pressure
        if not blood_pressure_med_treatment:  # untreated
            if systolic_blood_pressure < 120:
                points += 0
            if 120 <= systolic_blood_pressure <= 129:
                points += 1
            if 130 <= systolic_blood_pressure <= 139:
                points += 2
            if 140 <= systolic_blood_pressure <= 159:
                points += 3
            if systolic_blood_pressure >= 160:
                points += 4
        else:  # if the patient is on blood pressure meds
            if systolic_blood_pressure < 120:
                points += 0
            if 120 <= systolic_blood_pressure <= 129:
                points += 3
            if 130 <= systolic_blood_pressure <= 139:
                points += 4
            if 140 <= systolic_blood_pressure <= 159:
                points += 5
            if systolic_blood_pressure >= 160:
                points += 6

        # calulate % risk for females
        if points <= 9:
            percent_risk = 0.1
        elif 9 <= points <= 12:
            percent_risk = 1
        elif 13 <= points <= 14:
            percent_risk = 2
        elif points == 15:
            percent_risk = 3
        elif points == 16:
            percent_risk = 4
        elif points == 17:
            percent_risk = 5
        elif points == 18:
            percent_risk = 6
        elif points == 19:
            percent_risk = 8
        elif points == 20:
            percent_risk = 11
        elif points == 21:
            percent_risk = 14
        elif points == 22:
            percent_risk = 17
        elif points == 23:
            percent_risk = 22
        elif points == 24:
            percent_risk = 27
        elif points >= 25:
            percent_risk = 30

    return percent_risk / 100.0


def mmolL_to_mgdl(val: float) -> float:
    return val * 18.0182


class FraminghamModel:
    def __init__(self) -> None:
        pass

    def fit(self, *args: Any, **kwargs: Any) -> "FraminghamModel":
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
        def fram_inference(row: pd.DataFrame) -> pd.DataFrame:
            tchol = mmolL_to_mgdl(row["tchol"])
            hdlc = mmolL_to_mgdl(row["hdl"])

            score = inference(
                sex=row["sex"],
                age=row["age"],
                total_cholesterol=tchol,
                hdl_cholesterol=hdlc,
                systolic_blood_pressure=row["sbp"],
                smoker=row["smoker"],
                blood_pressure_med_treatment=row["ht_treat"],
            )
            return score

        scores = df.apply(lambda row: fram_inference(row), axis=1)

        return pd.DataFrame(scores, columns=[10 * 365])
