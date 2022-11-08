# stdlib
import os
from pathlib import Path

# third party
from helpers import MockHook
from lifelines.datasets import load_rossi
import numpy as np
import pytest

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.studies.risk_estimation import RiskEstimationStudy
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_survival_estimator


@pytest.mark.slow
def test_surv_search() -> None:
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.25)),
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    storage_folder = Path("/tmp")
    study_name = "test_risk_estimation_studies"
    workspace = storage_folder / study_name
    output = workspace / "model.p"

    try:
        os.remove(output)
    except OSError:
        pass

    study = RiskEstimationStudy(
        study_name=study_name,
        dataset=rossi,
        target="arrest",
        time_to_event="week",
        time_horizons=eval_time_horizons,
        num_iter=2,
        timeout=10,
        risk_estimators=["cox_ph", "lognormal_aft", "loglogistic_aft"],
        score_threshold=0.4,
        workspace=storage_folder,
    )

    study.run()

    assert output.is_file()

    model_v1 = load_model_from_file(output)

    metrics = evaluate_survival_estimator(
        model_v1, X, T.values, Y.values.tolist(), eval_time_horizons
    )
    score_v1 = metrics["clf"]["c_index"][0]

    # resume the study - should get at least the same score
    study.run()

    assert output.is_file()

    model_v2 = load_model_from_file(output)

    metrics = evaluate_survival_estimator(model_v2, X, T, Y, eval_time_horizons)
    score_v2 = metrics["clf"]["c_index"][0]

    assert score_v2 >= score_v1


def test_hooks() -> None:
    hooks = MockHook()

    rossi = load_rossi()

    Y = rossi["arrest"]
    T = rossi["week"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.25)),
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    storage_folder = Path("/tmp")
    study_name = "test_risk_estimation_studies"
    output = storage_folder / study_name

    try:
        os.remove(output)
    except OSError:
        pass

    study = RiskEstimationStudy(
        study_name=study_name,
        dataset=rossi,
        target="arrest",
        time_to_event="week",
        time_horizons=eval_time_horizons,
        num_iter=2,
        timeout=10,
        risk_estimators=["cox_ph", "lognormal_aft", "loglogistic_aft"],
        workspace=storage_folder,
        score_threshold=0.4,
        hooks=hooks,
    )

    with pytest.raises(StudyCancelled):
        study.run()


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["categorical", "continuous"])
@pytest.mark.parametrize("imputers", [["mean", "median"], ["mean"]])
def test_study_imputation(scenario: str, imputers: list) -> None:
    rossi = load_rossi()

    Y = rossi["arrest"]
    T = rossi["week"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    storage_folder = Path("/tmp")
    study_name = "test_risk_estimation_studies"
    output = storage_folder / study_name

    try:
        os.remove(output)
    except OSError:
        pass

    if scenario == "continuous":
        rossi.loc[0, "paro"] = np.nan
    else:
        rossi["cat_column"] = "test"
        rossi.loc[rossi["paro"] == 1, "cat_column"] = "other"
        rossi.loc[0, "cat_column"] = np.nan

    study = RiskEstimationStudy(
        study_name=study_name,
        dataset=rossi,
        target="arrest",
        time_to_event="week",
        time_horizons=eval_time_horizons,
        num_iter=2,
        num_study_iter=2,
        timeout=10,
        risk_estimators=["cox_ph", "lognormal_aft"],
        imputers=imputers,
        score_threshold=0.4,
        workspace=storage_folder,
    )

    study.run()
