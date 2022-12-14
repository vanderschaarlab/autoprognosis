# stdlib
from typing import Optional

# third party
from explorers_mocks import MockHook
import numpy as np
import pandas as pd
from pycox import datasets
import pytest
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.risk_estimation import RiskEstimatorSeeker
from autoprognosis.plugins.prediction import Predictions
from autoprognosis.utils.metrics import (
    evaluate_skurv_brier_score,
    evaluate_skurv_c_index,
)
from autoprognosis.utils.tester import evaluate_survival_estimator


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_sanity(optimizer_type: str) -> None:
    sq = RiskEstimatorSeeker(
        study_name="test_risk_estimation",
        time_horizons=[2],
        num_iter=3,
        CV=5,
        top_k=2,
        timeout=10,
        optimizer_type=optimizer_type,
    )

    assert sq.time_horizons == [2]
    assert sq.num_iter == 3
    assert sq.CV == 5
    assert sq.top_k == 2
    assert sq.timeout == 10
    assert len(sq.estimators) > 0


@pytest.mark.parametrize("group_id", [False, True])
def test_search(group_id: Optional[bool]) -> None:
    df = datasets.gbsg.read_df()
    df = df[df["duration"] > 0]

    X = df.drop(["duration", "event"], axis=1)
    Y = df["event"]
    T = df["duration"]

    group_ids = None
    if group_id:
        group_ids = pd.Series(np.random.randint(0, 10, X.shape[0]))

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
    ]
    estimators = ["lognormal_aft", "loglogistic_aft"]
    sq = RiskEstimatorSeeker(
        study_name="test_risk_estimation",
        time_horizons=eval_time_horizons,
        num_iter=10,
        CV=2,
        top_k=3,
        timeout=10,
        estimators=estimators,
    )

    best_models = sq.search(X, T, Y, group_ids=group_ids)

    tr_X, te_X, tr_T, te_T, tr_Y, te_Y = train_test_split(
        X, T, Y, test_size=0.2, random_state=0
    )

    assert len(best_models) == len(eval_time_horizons)
    assert len(best_models[0]) == len(estimators)

    for models, eval_time in zip(best_models, eval_time_horizons):
        print("Evaluating time horizon ", eval_time)
        for model in models:
            model.fit(tr_X, tr_T, tr_Y)

            print(f"Eval time {eval_time} best model: ", model.name())

            y_pred = model.predict(te_X, [eval_time]).to_numpy()

            c_index = evaluate_skurv_c_index(tr_T, tr_Y, y_pred, te_T, te_Y, eval_time)
            assert c_index > 0.5

            brier = evaluate_skurv_brier_score(
                tr_T, tr_Y, y_pred, te_T, te_Y, eval_time
            )
            assert brier < 0.5


def test_eval_surv_estimator() -> None:
    predictions = Predictions(category="risk_estimation")
    estimator = predictions.get("cox_ph")

    df = datasets.gbsg.read_df()
    df = df[df["duration"] > 0]

    X = df.drop(["duration", "event"], axis=1)
    Y = df["event"]
    T = df["duration"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]
    evaluate_survival_estimator(estimator, X, T, Y, eval_time_horizons, 5)


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_hooks(optimizer_type: str) -> None:
    hooks = MockHook()

    df = datasets.gbsg.read_df()
    df = df[df["duration"] > 0]

    X = df.drop(["duration", "event"], axis=1)
    Y = df["event"]
    T = df["duration"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
    ]
    estimators = ["lognormal_aft", "loglogistic_aft"]
    sq = RiskEstimatorSeeker(
        study_name="test_risk_estimation",
        time_horizons=eval_time_horizons,
        num_iter=20,
        CV=5,
        top_k=3,
        timeout=10,
        estimators=estimators,
        hooks=hooks,
        optimizer_type=optimizer_type,
    )

    with pytest.raises(StudyCancelled):
        sq.search(X, T, Y)
