# third party
import pytest

# autoprognosis absolute
from autoprognosis.plugins.prediction.risk_estimation.benchmarks.diabetes.qdiabetes.model import (
    inference,
)


@pytest.mark.parametrize("model", ["A", "B", "C"])
def test_sanity(model) -> None:
    score = inference(
        model,
        gender="M",
        age=84,  # age value
        b_atypicalantipsy=1,  # bool, On atypical antipsychotic medication
        b_corticosteroids=1,  # Are you on regular steroid tablets?
        b_cvd=1,  # Have you had a heart attack, angina, stroke or TIA?
        b_gestdiab=0,  # Women: Do you have gestational diabetes ?
        b_learning=0,  # Learning disabilities?
        b_manicschiz=0,  # Manic depression or schizophrenia?
        b_pos=0,  # Do you have polycystic ovaries?
        b_statin=0,  # Are you on statins?
        b_treatedhyp=1,  # Do you have high blood pressure requiring treatment?
        bmi=34,  # Body mass index = kg/m^2
        ethrisk=1,  # ethnic risk
        fh_diab=1,  # Do immediate family (mother, father, brothers or sisters) have diabetes?
        hba1c=40,  # HBA1c (mmol/mol)
        smoke_cat=4,  # smoking category: non-smoker, ex-smoker, light-smoker(less than 10/), moderate                 smoker(10-      19), heavy smoker(20 or over)
        fbs=0.01,
        town=0,  # Townsend deprivation score
    )

    assert score <= 1
