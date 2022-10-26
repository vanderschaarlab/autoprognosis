# third party
from explorers_mocks import MockHook
from lifelines.datasets import load_rossi
import pytest
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.risk_estimation_combos import RiskEnsembleSeeker
from autoprognosis.plugins.prediction import Predictions
from autoprognosis.utils.metrics import (
    evaluate_skurv_brier_score,
    evaluate_skurv_c_index,
)


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_sanity(optimizer_type: str) -> None:
    sq = RiskEnsembleSeeker(
        study_name="test_risk_estimation",
        time_horizons=[2],
        num_iter=3,
        CV=5,
        ensemble_size=2,
        timeout=10,
        optimizer_type=optimizer_type,
    )

    assert sq.time_horizons == [2]
    assert sq.num_iter == 3
    assert sq.CV == 5
    assert sq.ensemble_size == 2
    assert sq.timeout == 10


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_search(optimizer_type: str) -> None:

    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
    ]
    sq = RiskEnsembleSeeker(
        study_name="test_risk_estimation",
        time_horizons=eval_time_horizons,
        num_iter=15,
        num_ensemble_iter=3,
        CV=3,
        ensemble_size=3,
        timeout=10,
        estimators=["lognormal_aft", "loglogistic_aft"],
        optimizer_type=optimizer_type,
    )

    ensemble = sq.search(X, T, Y)

    assert len(ensemble.weights) == len(eval_time_horizons)

    tr_X, te_X, tr_T, te_T, tr_Y, te_Y = train_test_split(
        X, T, Y, test_size=0.1, random_state=0
    )

    ensemble.fit(tr_X, tr_T, tr_Y)

    for idx, eval_time in enumerate(eval_time_horizons):
        print("Evaluating time horizon ", eval_time)

        y_pred = ensemble.predict(te_X, [eval_time]).to_numpy()

        c_index = evaluate_skurv_c_index(tr_T, tr_Y, y_pred, te_T, te_Y, eval_time)
        assert c_index > 0

        brier = evaluate_skurv_brier_score(tr_T, tr_Y, y_pred, te_T, te_Y, eval_time)
        assert brier < 1

    for e_idx, eval_time in enumerate(eval_time_horizons):
        ind_est = (
            Predictions(category="risk_estimation").get("cox_ph").fit(tr_X, tr_T, tr_Y)
        )
        ind_pred = ind_est.predict(te_X, [eval_time]).to_numpy()
        ens_pred = ensemble.predict(te_X, [eval_time]).to_numpy()

        ind_c_index = evaluate_skurv_c_index(
            tr_T, tr_Y, ind_pred, te_T, te_Y, eval_time
        )
        ens_c_index = evaluate_skurv_c_index(
            tr_T, tr_Y, ens_pred, te_T, te_Y, eval_time
        )

        ind_brier = evaluate_skurv_brier_score(
            tr_T, tr_Y, ind_pred, te_T, te_Y, eval_time
        )
        ens_brier = evaluate_skurv_brier_score(
            tr_T, tr_Y, ens_pred, te_T, te_Y, eval_time
        )

        print(
            f"Comparing individual c_index {ind_c_index} with ensemble c_index {ens_c_index}"
        )
        print(
            f"Comparing individual brier_score {ind_brier} with ensemble c_index {ens_brier}"
        )

        assert (
            ind_c_index <= ens_c_index
        ), f"The ensemble should have a better c_index. horizon {eval_time}"


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_hooks(optimizer_type: str) -> None:
    hooks = MockHook()

    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
    ]
    sq = RiskEnsembleSeeker(
        study_name="test_risk_estimation",
        time_horizons=eval_time_horizons,
        num_iter=15,
        num_ensemble_iter=3,
        CV=3,
        ensemble_size=3,
        timeout=10,
        estimators=["lognormal_aft", "loglogistic_aft"],
        hooks=hooks,
        optimizer_type=optimizer_type,
    )

    with pytest.raises(StudyCancelled):
        sq.search(X, T, Y)
