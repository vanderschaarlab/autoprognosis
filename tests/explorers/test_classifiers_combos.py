# stdlib
import sys
from typing import Optional

import numpy as np
import pandas as pd
import pytest

# third party
from explorers_mocks import MockHook
from sklearn.datasets import load_breast_cancer

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.classifiers_combos import EnsembleSeeker
from autoprognosis.utils.metrics import evaluate_auc


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_sanity(optimizer_type: str) -> None:
    eseeker = EnsembleSeeker(
        study_name="test_classifiers_combos",
        n_folds_cv=10,
        num_iter=123,
        metric="aucprc",
        ensemble_size=12,
        feature_scaling=["scaler"],
        classifiers=["perceptron"],
        optimizer_type=optimizer_type,
    )

    assert eseeker.seeker.n_folds_cv == 10
    assert eseeker.seeker.num_iter == 123
    assert eseeker.ensemble_size == 1

    assert eseeker.seeker.estimators[0].feature_scaling[0].name() == "scaler"
    assert eseeker.seeker.estimators[0].name() == "perceptron"


def test_fails() -> None:
    with pytest.raises(ValueError):
        EnsembleSeeker(study_name="test_classifiers_combos", classifiers=["invalid"])

    with pytest.raises(ValueError):
        EnsembleSeeker(
            study_name="test_classifiers_combos", feature_scaling=["invalid"]
        )

    with pytest.raises(ValueError):
        EnsembleSeeker(study_name="test_classifiers_combos", n_folds_cv=-1)

    with pytest.raises(ValueError):
        EnsembleSeeker(study_name="test_classifiers_combos", num_iter=-2)

    with pytest.raises(ValueError):
        EnsembleSeeker(study_name="test_classifiers_combos", ensemble_size=-2)

    with pytest.raises(ValueError):
        EnsembleSeeker(study_name="test_classifiers_combos", metric="invalid")


@pytest.mark.skipif(sys.platform == "darwin", reason="slow")
@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
@pytest.mark.parametrize("group_id", [False, True])
def test_search(optimizer_type: str, group_id: Optional[bool]) -> None:
    X, Y = load_breast_cancer(return_X_y=True, as_frame=True)
    group_ids = None
    if group_id:
        group_ids = pd.Series(np.random.randint(0, 10, X.shape[0]))

    seeker = EnsembleSeeker(
        study_name="test_classifiers_combos",
        num_iter=2,
        num_ensemble_iter=3,
        feature_scaling=["scaler", "minmax_scaler"],
        classifiers=[
            "lda",
            "catboost",
        ],
        optimizer_type=optimizer_type,
    )

    selected_ensemble = seeker.search(X, Y, group_ids=group_ids)

    print("Best model ", selected_ensemble.name())
    selected_ensemble.fit(X, Y)

    y_pred_proba = selected_ensemble.predict_proba(X)

    assert len(y_pred_proba) == len(Y)

    assert evaluate_auc(Y, y_pred_proba)[0] > 0.9


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_hooks(optimizer_type: str) -> None:
    hook = MockHook()
    X, Y = load_breast_cancer(return_X_y=True, as_frame=True)

    seeker = EnsembleSeeker(
        study_name="test_classifiers_combos",
        num_iter=2,
        num_ensemble_iter=3,
        feature_scaling=["scaler", "minmax_scaler"],
        classifiers=[
            "lda",
            "catboost",
        ],
        hooks=hook,
        optimizer_type=optimizer_type,
    )

    with pytest.raises(StudyCancelled):
        seeker.search(X, Y)
