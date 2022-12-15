# stdlib
from typing import Any

# third party
from packaging import version
import sklearn
from sklearn.calibration import CalibratedClassifierCV

# autoprognosis absolute
from autoprognosis.utils.parallel import n_learner_jobs

calibrations = ["none", "sigmoid", "isotonic"]


def calibrated_model(model: Any, calibration: int = 1, **kwargs: Any) -> Any:
    if calibration >= len(calibrations):
        raise RuntimeError("invalid calibration value")

    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        est_kwargs = {
            "estimator": model,
        }
    else:
        est_kwargs = {
            "base_estimator": model,
        }
    if not hasattr(model, "predict_proba"):
        return CalibratedClassifierCV(**est_kwargs, n_jobs=n_learner_jobs())

    if calibration != 0:
        return CalibratedClassifierCV(
            **est_kwargs,
            method=calibrations[calibration],
            n_jobs=n_learner_jobs(),
        )

    return model
