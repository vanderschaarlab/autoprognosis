# stdlib
from typing import Optional

# third party
from explorers_mocks import MockHook
import numpy as np
import pytest
from sklearn.datasets import load_diabetes

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.regression_combos import RegressionEnsembleSeeker


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_sanity(optimizer_type: str) -> None:
    eseeker = RegressionEnsembleSeeker(
        study_name="test_regressors_combos",
        CV=10,
        num_iter=123,
        metric="r2",
        ensemble_size=12,
        feature_scaling=["scaler"],
        regressors=["linear_regression"],
        optimizer_type=optimizer_type,
    )

    assert eseeker.seeker.CV == 10
    assert eseeker.seeker.num_iter == 123
    assert eseeker.ensemble_size == 12

    assert eseeker.seeker.estimators[0].feature_scaling[0].name() == "scaler"
    assert eseeker.seeker.estimators[0].name() == "linear_regression"


def test_fails() -> None:
    with pytest.raises(ValueError):
        RegressionEnsembleSeeker(
            study_name="test_regressors_combos", regressors=["invalid"]
        )

    with pytest.raises(ValueError):
        RegressionEnsembleSeeker(
            study_name="test_regressors_combos", feature_scaling=["invalid"]
        )

    with pytest.raises(ValueError):
        RegressionEnsembleSeeker(study_name="test_regressors_combos", CV=-1)

    with pytest.raises(ValueError):
        RegressionEnsembleSeeker(study_name="test_regressors_combos", num_iter=-2)

    with pytest.raises(ValueError):
        RegressionEnsembleSeeker(study_name="test_regressors_combos", ensemble_size=-2)

    with pytest.raises(ValueError):
        RegressionEnsembleSeeker(study_name="test_regressors_combos", metric="invalid")

    with pytest.raises(ValueError):
        RegressionEnsembleSeeker(study_name="test_regressors_combos", metric="aucroc")


@pytest.mark.parametrize("group_id", [None, "id"])
def test_search(group_id: Optional[str]) -> None:
    X, Y = load_diabetes(return_X_y=True, as_frame=True)
    if group_id is not None:
        X[group_id] = np.random.randint(0, 10, size=(X.shape[0], 1))

    seeker = RegressionEnsembleSeeker(
        study_name="test_regressors_combos",
        num_iter=10,
        num_ensemble_iter=3,
        feature_scaling=["scaler", "minmax_scaler"],
        regressors=[
            "linear_regression",
            "random_forest_regressor",
        ],
    )

    selected_ensemble = seeker.search(X, Y, group_id=group_id)

    print("Best model ", selected_ensemble.name())
    selected_ensemble.fit(X, Y)

    y_pred = selected_ensemble.predict(X)

    assert len(y_pred) == len(Y)


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_hooks(optimizer_type: str) -> None:
    hook = MockHook()
    X, Y = load_diabetes(return_X_y=True, as_frame=True)

    seeker = RegressionEnsembleSeeker(
        study_name="test_regressors_combos",
        num_iter=10,
        num_ensemble_iter=3,
        feature_scaling=["scaler", "minmax_scaler"],
        regressors=[
            "linear_regression",
            "random_forest_regressor",
        ],
        hooks=hook,
        optimizer_type=optimizer_type,
    )

    with pytest.raises(StudyCancelled):
        seeker.search(X, Y)
