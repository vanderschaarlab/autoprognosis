# stdlib
from typing import Any, Tuple

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.plugins.prediction.risk_estimation.benchmarks.cvd.aha.model import (
    AHAModel,
)
from autoprognosis.plugins.prediction.risk_estimation.benchmarks.cvd.framingham.model import (
    FraminghamModel,
)
from autoprognosis.plugins.prediction.risk_estimation.benchmarks.cvd.qrisk3.model import (
    QRisk3Model,
)


def extras_cbk(raw_df: pd.DataFrame) -> Tuple[str, Any]:
    models = {
        "AHA/ACC score": eval_aha,
        "Framingham score": eval_fram,
        "QRisk3 score": eval_qrisk3,
    }

    results = pd.DataFrame(
        np.zeros((1, len(models))), columns=models.keys(), index=["10-year risk"]
    )

    for idx, reason in enumerate(models):
        predictions = models[reason](raw_df)
        results[reason] = np.round(predictions, 4)

    styles = [
        dict(selector="th", props=[("font-size", "18pt"), ("text-align", "center")]),
        dict(selector="tr", props=[("font-size", "16pt"), ("text-align", "center")]),
    ]

    results_styler = results.style.set_table_styles(styles)

    return ("table", results_styler)


def eval_qrisk3(raw_df: pd.DataFrame) -> float:
    gender = raw_df["Sex"]
    gender = gender.replace("Male", "M")
    gender = gender.replace("Female", "F")

    age = raw_df["Age"]

    b_AF = raw_df["Atrial fibrillation"]

    b_atypicalantipsy = raw_df["Drug status: atypical antipsychotic medication"]

    b_corticosteroids = raw_df["Drug status: steroid tablets"]

    b_migraine = raw_df["Do you have migraines?"]

    b_ra = raw_df["Rheumatoid arthritis"]

    b_renal = raw_df["Chronic kidney disease"]

    b_semi = raw_df["Severe mental illness?"]

    b_sle = raw_df["Systemic lupus erythematosus (SLE)"]

    b_treatedhyp = raw_df["Drug status: Anti-hypertensives"]

    b_type2 = raw_df["History of diabetes"]

    height_m = raw_df["Height (cm)"] / 100.0
    bmi = raw_df["Weight (kg)"] / (height_m * height_m)
    bmi.name = "bmi"

    ethrisk = pd.Series(np.zeros(len(raw_df)), name="ethrisk", index=age.index)  #
    ethrisk[raw_df["Ethnicity"] == "Black"] = 6
    ethrisk[raw_df["Ethnicity"] == "Asian / Oriental"] = 4
    ethrisk[raw_df["Ethnicity"] == "Other"] = 8
    ethrisk = ethrisk.astype(int)

    rati = raw_df["Total cholesterol"]  # Cholesterol/HDL ratio
    rati /= raw_df["HDL"]
    rati.name = "chol_ratio"

    sbp = raw_df["SBP (mmHg)"]

    sbps5 = pd.Series(np.zeros(len(raw_df)), name="sbps5", index=age.index)  # ??

    smoke_cat = (raw_df["Smoking amount (cigarettes/day)"] > 0).astype(int)
    smoke_cat = smoke_cat.replace(1, 2)

    qrisk_input = pd.concat(
        [
            gender,  # M/F
            age,  # age value
            b_AF,  # bool, Atrial fibrillation
            b_atypicalantipsy,  # bool, On atypical antipsychotic medication
            b_corticosteroids,  # Are you on regular steroid tablets?
            b_migraine,  # bool, Do you have migraines?
            b_ra,  # Rheumatoid arthritis?
            b_renal,  # Chronic kidney disease (stage 3, 4 or 5)?
            b_semi,  # Severe mental illness?
            b_sle,  # Systemic lupus erythematosus
            b_treatedhyp,  # On blood pressure treatment?
            b_type2,  # Diabetes status: type 2
            bmi,  # Body mass index = kg/m^2
            ethrisk,  # ethnic risk
            rati,  # Cholesterol/HDL ratio
            sbp,  # Systolic blood pressure
            sbps5,  # Standard deviation of at least two most recent systolic blood pressure readings (mmHg)
            smoke_cat,  # smoking category: non-smoker, ex-smoker, light-smoker(less than 10/), moderate                 smoker(10-      19), heavy smoker(20 or over)
        ],
        axis=1,
    )
    # missing in biobank
    qrisk_input["sbps5"] = 0
    qrisk_input["town_depr_index"] = 0
    qrisk_input["family_cvd"] = 0
    qrisk_input["b_erectile_disf"] = 0
    qrisk_input["b_diab_type1"] = 0

    qrisk_input = qrisk_input.rename(
        columns={
            "Sex": "sex",
            "Age": "age",
            "Atrial fibrillation": "b_atrial_fibr",
            "Drug status: atypical antipsychotic medication": "b_antipsychotic_use",
            "Drug status: steroid tablets": "b_steroid_treat",
            "Do you have migraines?": "b_had_migraine",
            "Rheumatoid arthritis": "b_rheumatoid_arthritis",
            "Chronic kidney disease": "b_renal",
            "Severe mental illness?": "b_mental_illness",
            "Systemic lupus erythematosus (SLE)": "b_sle",
            "Drug status: Anti-hypertensives": "ht_treat",
            "History of diabetes": "b_diab_type2",
            "SBP (mmHg)": "sbp",
            "Smoking amount (cigarettes/day)": "smoker",
        }
    )

    pred = QRisk3Model().predict(qrisk_input).values.squeeze()

    return pred


def eval_aha(raw_df: pd.DataFrame) -> float:
    gender = raw_df["Sex"]
    gender = gender.replace("Male", "M")
    gender = gender.replace("Female", "F")

    age = raw_df["Age"]

    tchol = raw_df["Total cholesterol"]

    hdlc = raw_df["HDL"]

    sbp = raw_df["SBP (mmHg)"]

    smoking = (raw_df["Smoking amount (cigarettes/day)"] > 0).astype(int)

    diab = raw_df["History of diabetes"]

    ht_treat = raw_df["Drug status: Anti-hypertensives"]

    race = (raw_df["Ethnicity"] == "Black").astype(int)  #
    race = race.replace(1, "B")
    race = race.replace(0, "W")

    aha_input = pd.concat(
        [gender, age, tchol, hdlc, sbp, smoking, diab, ht_treat, race], axis=1
    )
    aha_input = aha_input.rename(
        columns={
            "Sex": "sex",
            "Age": "age",
            "Total cholesterol": "tchol",
            "HDL": "hdl",
            "SBP (mmHg)": "sbp",
            "Smoking amount (cigarettes/day)": "smoker",
            "History of diabetes": "diabetes",
            "Drug status: Anti-hypertensives": "ht_treat",
            "Ethnicity": "race",
        }
    )

    pred = AHAModel().predict(aha_input).values.squeeze()
    return pred


def eval_fram(raw_df: pd.DataFrame) -> float:
    gender = raw_df["Sex"]
    gender = gender.replace("Male", "M")
    gender = gender.replace("Female", "F")

    age = raw_df["Age"]

    tchol = raw_df["Total cholesterol"]

    hdlc = raw_df["HDL"]

    sbp = raw_df["SBP (mmHg)"]

    smoking = (raw_df["Smoking amount (cigarettes/day)"] > 0).astype(int)

    ht_treat = raw_df["Drug status: Anti-hypertensives"]

    fram_input = pd.concat([gender, age, tchol, hdlc, sbp, ht_treat, smoking], axis=1)

    fram_input = fram_input.rename(
        columns={
            "Sex": "sex",
            "Age": "age",
            "Total cholesterol": "tchol",
            "HDL": "hdl",
            "SBP (mmHg)": "sbp",
            "Smoking amount (cigarettes/day)": "smoker",
            "History of diabetes": "diabetes",
            "Drug status: Anti-hypertensives": "ht_treat",
            "Ethnicity": "race",
        }
    )

    pred = FraminghamModel().predict(fram_input).values.squeeze()

    return pred
