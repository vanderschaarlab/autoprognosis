# stdlib
from typing import Any

# third party
from sklearn.calibration import CalibratedClassifierCV

# autoprognosis absolute
from autoprognosis.utils.parallel import n_learner_jobs

calibrations = ["none", "sigmoid", "isotonic"]


def calibrated_model(model: Any, calibration: int = 1, **kwargs: Any) -> Any:
    if calibration >= len(calibrations):
        raise RuntimeError("invalid calibration value")

    if not hasattr(model, "predict_proba"):
        return CalibratedClassifierCV(base_estimator=model, n_jobs=n_learner_jobs())

    if calibration != 0:
        return CalibratedClassifierCV(
            base_estimator=model,
            method=calibrations[calibration],
            n_jobs=n_learner_jobs(),
        )

    return model
