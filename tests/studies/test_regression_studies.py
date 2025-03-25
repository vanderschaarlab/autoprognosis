# stdlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# third party
from helpers import MockHook
from sklearn.datasets import load_diabetes

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.studies.regression import RegressionStudy
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_regression


@pytest.mark.skipif(sys.platform != "linux", reason="slow")
@pytest.mark.parametrize("sample_for_search", [True, False])
def test_regression_search(sample_for_search: bool) -> None:
    X, Y = load_diabetes(return_X_y=True, as_frame=True)

    df = X.copy()
    df["target"] = Y

    storage_folder = Path("/tmp")
    study_name = "test_regressors_studies"
    workspace = storage_folder / study_name
    output = workspace / "model.p"
    try:
        os.remove(output)
    except OSError:
        pass

    study = RegressionStudy(
        study_name=study_name,
        dataset=df,
        target="target",
        num_iter=2,
        num_study_iter=1,
        timeout=10,
        regressors=["linear_regression"],
        workspace=storage_folder,
        score_threshold=0.3,
        sample_for_search=sample_for_search,
    )
    print("study name", study.study_name)

    study.run()

    assert output.is_file()

    model_v1 = load_model_from_file(output)
    assert not model_v1.is_fitted()

    metrics = evaluate_regression(model_v1, X, Y)
    score_v1 = metrics["raw"]["r2"][0]

    # resume the study - should get at least the same score
    study.run()

    assert output.is_file()

    model_v2 = load_model_from_file(output)

    metrics = evaluate_regression(model_v2, X, Y)
    score_v2 = metrics["raw"]["r2"][0]

    EPS = 0.05
    assert score_v2 + EPS >= score_v1

    model = study.fit()
    assert model.is_fitted()

    preds = model.predict(X)
    assert len(preds) == len(X)


def test_hooks() -> None:
    hooks = MockHook()
    X, Y = load_diabetes(return_X_y=True, as_frame=True)

    df = X.copy()
    df["target"] = Y

    storage_folder = Path("/tmp")
    study_name = "test_regressors_studies"
    output = storage_folder / study_name

    try:
        os.remove(output)
    except OSError:
        pass

    study = RegressionStudy(
        study_name=study_name,
        dataset=df,
        target="target",
        num_iter=10,
        num_study_iter=3,
        timeout=10,
        regressors=["linear_regression", "random_forest_regressor"],
        workspace=storage_folder,
        hooks=hooks,
    )
    with pytest.raises(StudyCancelled):
        study.run()


@pytest.mark.parametrize("imputers", [["mean", "median"], ["mean"]])
@pytest.mark.skipif(sys.platform != "linux", reason="slow")
def test_regression_study_imputation(imputers: list) -> None:
    X, Y = load_diabetes(return_X_y=True, as_frame=True)
    storage_folder = Path("/tmp")
    study_name = "test_regressors_studies"
    workspace = storage_folder / study_name
    output = workspace / "model.p"
    try:
        os.remove(output)
    except OSError:
        pass

    X.loc[0, "mean smoothness"] = np.nan

    df = X.copy()
    df["target"] = Y

    study = RegressionStudy(
        study_name=study_name,
        dataset=df,
        target="target",
        imputers=imputers,
        num_iter=2,
        num_study_iter=1,
        timeout=10,
        regressors=["linear_regression"],
        workspace=storage_folder,
        score_threshold=0.3,
    )

    study.run()

    assert output.is_file()
