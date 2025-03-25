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
from autoprognosis.explorers.classifiers import ClassifierSeeker
from autoprognosis.utils.metrics import evaluate_auc


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_sanity(optimizer_type: str) -> None:
    model = ClassifierSeeker(
        study_name="test_classifiers",
        n_folds_cv=10,
        num_iter=123,
        top_k=5,
        timeout=6,
        metric="aucprc",
        feature_scaling=["scaler"],
        classifiers=["perceptron"],
        optimizer_type=optimizer_type,
    )

    assert model.n_folds_cv == 10
    assert model.num_iter == 123
    assert model.top_k == 5
    assert model.timeout == 6

    assert model.estimators[0].feature_scaling[0].name() == "scaler"
    assert model.estimators[0].name() == "perceptron"


def test_fails() -> None:
    with pytest.raises(ValueError):
        ClassifierSeeker(study_name="test_classifiers", classifiers=["invalid"])

    with pytest.raises(ValueError):
        ClassifierSeeker(study_name="test_classifiers", feature_scaling=["invalid"])

    with pytest.raises(ValueError):
        ClassifierSeeker(study_name="test_classifiers", n_folds_cv=-1)

    with pytest.raises(ValueError):
        ClassifierSeeker(study_name="test_classifiers", num_iter=-2)

    with pytest.raises(ValueError):
        ClassifierSeeker(study_name="test_classifiers", metric="invalid")


@pytest.mark.skipif(sys.platform == "darwin", reason="slow")
@pytest.mark.parametrize(
    "optimizer_type,group_id",
    [("bayesian", False), ("bayesian", True), ("hyperband", False)],
)
@pytest.mark.parametrize("metric", ["f1_score_micro", "aucroc"])
def test_search_clf(optimizer_type: str, group_id: Optional[bool], metric: str) -> None:
    X, Y = load_breast_cancer(return_X_y=True, as_frame=True)
    group_ids = None
    if group_id:
        group_ids = pd.Series(np.random.randint(0, 10, X.shape[0]))

    seeker = ClassifierSeeker(
        study_name="test_classifiers",
        num_iter=2,
        top_k=3,
        feature_scaling=["scaler", "minmax_scaler"],
        classifiers=[
            "lda",
            "catboost",
        ],
        optimizer_type=optimizer_type,
        metric=metric,
        strict=True,
    )
    best_models = seeker.search(X, Y, group_ids=group_ids)

    assert len(best_models) == 2

    for model in best_models:
        model.fit(X, Y)

        y_pred = model.predict(X)
        y_pred_orig = model.predict(X)

        assert len(y_pred) == len(y_pred_orig)
        assert len(y_pred) == len(Y)

        y_pred_proba = model.predict_proba(X)

        assert len(y_pred_proba) == len(Y)

        assert evaluate_auc(Y, y_pred_proba)[0] > 0.9


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_hooks(optimizer_type: str) -> None:
    hook = MockHook()

    X, Y = load_breast_cancer(return_X_y=True, as_frame=True)

    seeker = ClassifierSeeker(
        study_name="test_classifiers",
        num_iter=2,
        top_k=3,
        hooks=hook,
        classifiers=["lda", "catboost"],
        optimizer_type=optimizer_type,
    )

    with pytest.raises(StudyCancelled):
        seeker.search(X, Y)
