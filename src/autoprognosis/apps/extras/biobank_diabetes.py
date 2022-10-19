# stdlib
from typing import Any, Tuple

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.plugins.prediction.risk_estimation.benchmarks.diabetes.ada.model import (
    ADAModel,
)
from autoprognosis.plugins.prediction.risk_estimation.benchmarks.diabetes.diabetes_uk.model import (
    DiabetesUKModel,
)
from autoprognosis.plugins.prediction.risk_estimation.benchmarks.diabetes.finrisk.model import (
    FINRISKModel,
)
from autoprognosis.plugins.prediction.risk_estimation.benchmarks.diabetes.qdiabetes.model import (
    QDiabetesModel,
)


def extras_cbk(raw_df: pd.DataFrame) -> Tuple[str, Any]:
    models = {
        "ADA score": eval_ada,
        "FINRISK": eval_finrisk,
        "DiabetesUK": eval_diabetesuk,
        "QDiabetes Model A": eval_qdiabetes_model_a,
        "QDiabetes Model B": eval_qdiabetes_model_b,
        "QDiabetes Model C": eval_qdiabetes_model_c,
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


def eval_ada(raw_df: pd.DataFrame) -> float:
    gender = raw_df["Sex"]
    gender = gender.replace("Male", "M")
    gender = gender.replace("Female", "F")

    age = raw_df["Age"]
    hdiab = raw_df["Family History of Diabetes"].astype(int)
    ht_treat = raw_df["Drug status: Anti-hypertensives"].astype(int)

    height_m = raw_df["Height (cm)"] / 100.0
    bmi = raw_df["Weight (kg)"] / (height_m * height_m)
    bmi.name = "bmi"

    daily_ex = pd.Series(
        np.zeros(len(raw_df)), name="b_daily_exercise", index=age.index
    )

    ada_input = pd.concat([gender, age, hdiab, ht_treat, bmi, daily_ex], axis=1)

    ada_input = ada_input.rename(
        columns={
            "Sex": "sex",
            "Age": "age",
            "Family History of Diabetes": "fh_diab",
            "Drug status: Anti-hypertensives": "ht_treat",
        }
    )

    ada_input = ada_input.reset_index(drop=True)

    return ADAModel().predict(ada_input).values.squeeze()


def eval_finrisk(raw_df: pd.DataFrame) -> float:
    gender = raw_df["Sex"]
    gender = gender.replace("Male", "M")
    gender = gender.replace("Female", "F")

    age = raw_df["Age"]

    height_m = raw_df["Height (cm)"] / 100.0
    bmi = raw_df["Weight (kg)"] / (height_m * height_m)
    bmi.name = "bmi"

    ht_treat = raw_df["Drug status: Anti-hypertensives"].astype(int)
    hdiab = raw_df["Family History of Diabetes"].astype(int)

    waist = raw_df["Waist (cm)"]

    finrisk_input = pd.concat([gender, age, bmi, ht_treat, waist, hdiab], axis=1)

    finrisk_input = finrisk_input.rename(
        columns={
            "Sex": "sex",
            "Age": "age",
            "History of diabetes": "fh_diab",
            "Family History of Diabetes": "fh_diab",
            "Drug status: Anti-hypertensives": "ht_treat",
            "Waist (cm)": "waist",
        }
    )

    return FINRISKModel().predict(finrisk_input).values.squeeze()


def eval_diabetesuk(raw_df: pd.DataFrame) -> float:
    gender = raw_df["Sex"]
    gender = gender.replace("Female", "M")
    gender = gender.replace("Male", "F")

    age = raw_df["Age"]

    height_m = raw_df["Height (cm)"] / 100.0
    bmi = raw_df["Weight (kg)"] / (height_m * height_m)
    bmi.name = "bmi"

    ht_treat = raw_df["Drug status: Anti-hypertensives"].astype(int)
    hdiab = raw_df["Family History of Diabetes"].astype(int)

    waist = raw_df["Waist (cm)"]

    ethrisk = pd.Series(np.zeros(len(raw_df)), name="ethrisk", index=age.index)  #
    ethrisk[raw_df["Ethnicity"] == "Black"] = 6
    ethrisk[raw_df["Ethnicity"] == "Asian / Oriental"] = 4
    ethrisk[raw_df["Ethnicity"] == "Other"] = 8
    ethrisk = ethrisk.astype(int)

    diabuk_input = pd.concat(
        [gender, age, bmi, ht_treat, waist, hdiab, ethrisk], axis=1
    )

    diabuk_input = diabuk_input.rename(
        columns={
            "Sex": "sex",
            "Age": "age",
            "History of diabetes": "fh_diab",
            "Family History of Diabetes": "fh_diab",
            "Drug status: Anti-hypertensives": "ht_treat",
            "Waist (cm)": "waist",
        }
    )

    return DiabetesUKModel().predict(diabuk_input).values.squeeze()


def hba1c_pct_to_mmol(val):
    vals = 10.93 * val - 23.5
    vals[vals < 0] = 1e-8
    return vals


def eval_qdiabetes(raw_df: pd.DataFrame, model_type: str) -> float:
    gender = raw_df["Sex"]
    gender = gender.replace("Male", "M")
    gender = gender.replace("Female", "F")

    age = raw_df["Age"]

    b_atypicalantipsy = raw_df["Drug status: atypical antipsychotic medication"]
    b_corticosteroids = raw_df["Drug status: steroid tablets"]

    height_m = raw_df["Height (cm)"] / 100.0
    bmi = raw_df["Weight (kg)"] / (height_m * height_m)
    bmi.name = "bmi"

    ht_treat = raw_df["Drug status: Anti-hypertensives"].astype(int)

    ethrisk = pd.Series(np.zeros(len(raw_df)), name="ethrisk", index=age.index)  #
    ethrisk[raw_df["Ethnicity"] == "Black"] = 6
    ethrisk[raw_df["Ethnicity"] == "Asian / Oriental"] = 4
    ethrisk[raw_df["Ethnicity"] == "Other"] = 8
    ethrisk = ethrisk.astype(int)

    hdiab = raw_df["Family History of Diabetes"].astype(int)
    hba1c = hba1c_pct_to_mmol(raw_df["HbA1c (%)"])

    smoke_cat = raw_df["Smoker"].astype(int)

    fbs = raw_df["Fasting blood glucose (mmol/l)"]

    # CVD history
    b_cvd = raw_df["History of CVD"].astype(int)
    b_cvd.name = "b_cvd"

    b_gestdiab = pd.Series(np.zeros(len(raw_df)), name="b_gestdiab", index=age.index)
    b_learning = pd.Series(np.zeros(len(raw_df)), name="b_learning", index=age.index)
    b_manicschiz = pd.Series(
        np.zeros(len(raw_df)), name="b_manicschiz", index=age.index
    )
    b_pos = pd.Series(np.zeros(len(raw_df)), name="b_pos", index=age.index)
    b_statin = pd.Series(np.zeros(len(raw_df)), name="b_statin", index=age.index)
    town = pd.Series(np.zeros(len(raw_df)), name="town_depr_index", index=age.index)
    qdiab_input = pd.concat(
        [
            gender,
            age,
            b_atypicalantipsy,
            b_corticosteroids,
            bmi,
            town,
            ht_treat,
            ethrisk,
            hdiab,
            hba1c,
            smoke_cat,
            b_cvd,
            b_gestdiab,
            b_learning,
            b_manicschiz,
            b_pos,
            b_statin,
            fbs,
        ],
        axis=1,
    )

    qdiab_input = qdiab_input.rename(
        columns={
            "Sex": "sex",
            "Age": "age",
            "History of diabetes": "fh_diab",
            "Family History of Diabetes": "fh_diab",
            "Drug status: Anti-hypertensives": "ht_treat",
            "Waist (cm)": "waist",
            "Fasting blood glucose (mmol/l)": "fbs",
            "Smoker": "smoker",
            "HbA1c (%)": "hba1c",
            "Drug status: atypical antipsychotic medication": "b_antipsychotic_use",
            "Drug status: steroid tablets": "b_steroid_treat",
        }
    )

    return QDiabetesModel(model_type).predict(qdiab_input).values.squeeze()


def eval_qdiabetes_model_a(raw_df: pd.DataFrame) -> float:
    return eval_qdiabetes(raw_df, "A")


def eval_qdiabetes_model_b(raw_df: pd.DataFrame) -> float:
    return eval_qdiabetes(raw_df, "B")


def eval_qdiabetes_model_c(raw_df: pd.DataFrame) -> float:
    return eval_qdiabetes(raw_df, "C")
