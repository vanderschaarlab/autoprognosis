# stdlib
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# third party
from explorers_mocks import MockHook
from sklearn.datasets import load_diabetes

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.regression_combos import RegressionEnsembleSeeker


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_sanity(optimizer_type: str) -> None:
    eseeker = RegressionEnsembleSeeker(
        study_name="test_regressors_combos",
        n_folds_cv=10,
        num_iter=123,
        metric="r2",
        ensemble_size=12,
        feature_scaling=["scaler"],
        regressors=["linear_regression"],
        optimizer_type=optimizer_type,
    )

    assert eseeker.seeker.n_folds_cv == 10
    assert eseeker.seeker.num_iter == 123
    assert eseeker.ensemble_size == 1

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
        RegressionEnsembleSeeker(study_name="test_regressors_combos", n_folds_cv=-1)

    with pytest.raises(ValueError):
        RegressionEnsembleSeeker(study_name="test_regressors_combos", num_iter=-2)

    with pytest.raises(ValueError):
        RegressionEnsembleSeeker(study_name="test_regressors_combos", ensemble_size=-2)

    with pytest.raises(ValueError):
        RegressionEnsembleSeeker(study_name="test_regressors_combos", metric="invalid")

    with pytest.raises(ValueError):
        RegressionEnsembleSeeker(study_name="test_regressors_combos", metric="aucroc")


@pytest.mark.parametrize("group_id", [False, True])
def test_search(group_id: Optional[bool]) -> None:
    X, Y = load_diabetes(return_X_y=True, as_frame=True)
    group_ids = None
    if group_id:
        group_ids = pd.Series(np.random.randint(0, 10, X.shape[0]))

    seeker = RegressionEnsembleSeeker(
        study_name="test_regressors_combos",
        num_iter=2,
        num_ensemble_iter=3,
        feature_scaling=["scaler", "minmax_scaler"],
        regressors=[
            "linear_regression",
            "xgboost_regressor",
        ],
    )

    selected_ensemble = seeker.search(X, Y, group_ids=group_ids)

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
        num_iter=20,
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
