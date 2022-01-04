# third party
from explorers_mocks import MockHook
from lifelines.datasets import load_rossi
import pytest
from sklearn.model_selection import train_test_split

# adjutorium absolute
from adjutorium.exceptions import StudyCancelled
from adjutorium.explorers.risk_estimation import RiskEstimatorSeeker
from adjutorium.plugins.prediction import Predictions
from adjutorium.utils.metrics import evaluate_skurv_brier_score, evaluate_skurv_c_index
from adjutorium.utils.tester import evaluate_survival_estimator


def test_sanity() -> None:
    sq = RiskEstimatorSeeker(
        study_name="test_risk_estimation",
        time_horizons=[2],
        num_iter=3,
        CV=5,
        top_k=2,
        timeout=10,
    )

    assert sq.time_horizons == [2]
    assert sq.num_iter == 3
    assert sq.CV == 5
    assert sq.top_k == 2
    assert sq.timeout == 10
    assert len(sq.estimators) > 0


@pytest.mark.slow
def test_search() -> None:

    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.25)),
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]
    sq = RiskEstimatorSeeker(
        study_name="test_risk_estimation",
        time_horizons=eval_time_horizons,
        num_iter=20,
        CV=5,
        top_k=3,
        timeout=10,
        estimators=["loglogistic_aft", "lognormal_aft", "cox_ph"],
    )

    best_models = sq.search(X, T, Y)

    tr_X, te_X, tr_T, te_T, tr_Y, te_Y = train_test_split(
        X, T, Y, test_size=0.2, random_state=0
    )

    assert len(best_models) == len(eval_time_horizons)
    assert len(best_models[0]) == 3

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

    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]
    evaluate_survival_estimator(estimator, X, T, Y, eval_time_horizons, 5)


def test_hooks() -> None:
    hooks = MockHook()
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.25)),
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]
    sq = RiskEstimatorSeeker(
        study_name="test_risk_estimation",
        time_horizons=eval_time_horizons,
        num_iter=20,
        CV=5,
        top_k=3,
        timeout=10,
        estimators=["loglogistic_aft", "lognormal_aft", "cox_ph"],
        hooks=hooks,
    )

    with pytest.raises(StudyCancelled):
        sq.search(X, T, Y)
