# adjutorium absolute
from adjutorium.plugins.prediction.risk_estimation.benchmarks.cvd.framingham.model import (
    inference,
)


def test_sanity() -> None:
    score = inference(
        sex="F",
        age=60,  # age value
        total_cholesterol=204,
        hdl_cholesterol=38.67,
        systolic_blood_pressure=160,  # Systolic blood pressure
        smoker=True,
        blood_pressure_med_treatment=True,
    )

    assert score < 1
