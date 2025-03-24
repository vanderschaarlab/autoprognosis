# stdlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# third party
from helpers import MockHook
from sklearn.datasets import load_breast_cancer

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.studies.classifiers import ClassifierStudy
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_estimator


@pytest.mark.skipif(sys.platform != "linux", reason="slow")
@pytest.mark.parametrize("sample_for_search", [True, False])
def test_search(sample_for_search: bool) -> None:
    X, Y = load_breast_cancer(return_X_y=True, as_frame=True)

    df = X.copy()
    df["target"] = Y

    storage_folder = Path("/tmp")
    study_name = "test_classifiers_studies"
    workspace = storage_folder / study_name
    output = workspace / "model.p"
    try:
        os.remove(output)
    except OSError:
        pass

    study = ClassifierStudy(
        study_name=study_name,
        dataset=df,
        target="target",
        num_iter=2,
        num_study_iter=1,
        timeout=10,
        classifiers=["catboost"],
        workspace=storage_folder,
        sample_for_search=sample_for_search,
    )
    print("study name", study.study_name)

    study.run()

    assert output.is_file()

    model_v1 = load_model_from_file(output)
    assert not model_v1.is_fitted()

    metrics = evaluate_estimator(model_v1, X, Y)
    score_v1 = metrics["raw"]["aucroc"][0]

    # resume the study - should get at least the same score
    study.run()

    assert output.is_file()

    model_v2 = load_model_from_file(output)

    metrics = evaluate_estimator(model_v2, X, Y)
    score_v2 = metrics["raw"]["aucroc"][0]

    EPS = 1e-2
    assert score_v2 + EPS >= score_v1

    model = study.fit()
    assert model.is_fitted()

    preds = model.predict(X)
    assert len(preds) == len(X)


def test_hooks() -> None:
    hooks = MockHook()
    X, Y = load_breast_cancer(return_X_y=True, as_frame=True)

    df = X.copy()
    df["target"] = Y

    storage_folder = Path("/tmp")
    study_name = "test_classifiers_studies"
    output = storage_folder / study_name

    try:
        os.remove(output)
    except OSError:
        pass

    study = ClassifierStudy(
        study_name=study_name,
        dataset=df,
        target="target",
        num_iter=4,
        timeout=10,
        classifiers=["logistic_regression", "lda", "qda"],
        workspace=storage_folder,
        hooks=hooks,
    )
    with pytest.raises(StudyCancelled):
        study.run()


@pytest.mark.parametrize("imputers", [["mean", "median"], ["mean"]])
@pytest.mark.skipif(sys.platform != "linux", reason="slow")
def test_classification_study_imputation(imputers: list) -> None:
    X, Y = load_breast_cancer(return_X_y=True, as_frame=True)
    storage_folder = Path("/tmp")
    study_name = "test_classifiers_studies"
    workspace = storage_folder / study_name
    output = workspace / "model.p"
    try:
        os.remove(output)
    except OSError:
        pass

    X.loc[0, "mean smoothness"] = np.nan

    df = X.copy()
    df["target"] = Y

    study = ClassifierStudy(
        study_name=study_name,
        dataset=df,
        target="target",
        imputers=imputers,
        num_iter=2,
        num_study_iter=1,
        timeout=10,
        classifiers=["logistic_regression"],
        workspace=storage_folder,
    )

    study.run()

    assert output.is_file()
