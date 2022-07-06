# adjutorium absolute
from adjutorium.plugins.prediction.risk_estimation.benchmarks.cvd.aha.model import (
    inference,
)


def test_sanity() -> None:
    score = inference(
        gender="M",
        age=40,
        tchol=160,
        hdlc=40,
        sbp=180,
        smoking=0,
        diab=0,
        ht_treat=1,
        race="W",
    )
    assert score < 1
