# adjutorium absolute
from adjutorium.plugins.prediction.risk_estimation.benchmarks.diabetes.diabetes_uk.model import (
    inference,
)


def test_sanity() -> None:
    score = inference(
        gender="F",
        age=64,  # age value
        ethrisk=0,  # ethnic risk
        fh_diab=0,  # Do immediate family (mother, father, brothers or sisters) have diabetes?
        waist=80,
        bmi=24,  # Body mass index = kg/m^2
        b_treatedhyp=1,  # Do you have high blood pressure requiring treatment?
    )

    assert score < 1
