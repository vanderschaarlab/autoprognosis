# third party
from lifelines.datasets import load_rossi
import numpy as np
from sklearn.model_selection import train_test_split

# adjutorium absolute
from adjutorium.plugins.ensemble.risk_estimation import RiskEnsemble
from adjutorium.plugins.prediction import Predictions
from adjutorium.utils.metrics import (
    evaluate_skurv_brier_score,
    evaluate_skurv_c_index,
    evaluate_weighted_brier_score,
    evaluate_weighted_c_index,
)


def test_risk_estimation_ensemble_predict() -> None:
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    tr_X, te_X, tr_T, te_T, tr_Y, te_Y = train_test_split(
        X, T, Y, test_size=0.1, random_state=0
    )

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.25)),
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    cox_ph = Predictions(category="risk_estimation").get("cox_ph")
    survival_xgboost = Predictions(category="risk_estimation").get("survival_xgboost")
    lognormal_aft = Predictions(category="risk_estimation").get("lognormal_aft")

    weights = np.asarray(
        [
            [0.99, 0, 0],
            [0, 0.49, 0.499],
            [0.99, 1, 1],
        ]
    )
    surv_ensemble = RiskEnsemble(
        [cox_ph, survival_xgboost, lognormal_aft],
        weights,
        eval_time_horizons,
    )

    for test in [
        Predictions(category="risk_estimation").get("loglogistic_aft"),
        Predictions(category="risk_estimation").get("cox_ph"),
        surv_ensemble,
    ]:
        test.fit(tr_X, tr_T, tr_Y)

        for e_idx, eval_time in enumerate(eval_time_horizons):
            y_pred = test.predict(te_X, [eval_time]).to_numpy()

            c_index = evaluate_weighted_c_index(
                tr_T, tr_Y, y_pred, te_T, te_Y, eval_time
            )
            assert c_index > 0.5

            c_index = evaluate_skurv_c_index(tr_T, tr_Y, y_pred, te_T, te_Y, eval_time)
            assert c_index > 0.5

            brier = evaluate_weighted_brier_score(
                tr_T, tr_Y, y_pred, te_T, te_Y, eval_time
            )
            assert brier < 0.5

            brier = evaluate_skurv_brier_score(
                tr_T, tr_Y, y_pred, te_T, te_Y, eval_time
            )
            assert brier < 0.5
            print("perf metrics ", test.name(), c_index, brier)

    for e_idx, eval_time in enumerate(eval_time_horizons):
        ind_est = (
            Predictions(category="risk_estimation").get("cox_ph").fit(tr_X, tr_T, tr_Y)
        )
        ens_est = surv_ensemble.fit(tr_X, tr_T, tr_Y)

        ind_pred = ind_est.predict(te_X, [eval_time]).to_numpy()
        ens_pred = ens_est.predict(te_X, [eval_time]).to_numpy()

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
            f"[{e_idx}] Comparing individual c_index {ind_c_index} with ensemble c_index {ens_c_index}"
        )
        print(
            f"[{e_idx}] Comparing individual brier_score {ind_brier} with ensemble c_index {ens_brier}"
        )

        assert (
            ind_c_index <= ens_c_index
        ), f"The ensemble should have a better c_index. horizon {eval_time}"


def test_risk_estimation_explain() -> None:
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    tr_X, te_X, tr_T, te_T, tr_Y, te_Y = train_test_split(
        X, T, Y, test_size=0.1, random_state=0
    )

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.25)),
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    cox_ph = Predictions(category="risk_estimation").get("cox_ph")
    survival_xgboost = Predictions(category="risk_estimation").get("survival_xgboost")
    lognormal_aft = Predictions(category="risk_estimation").get("lognormal_aft")

    weights = np.asarray(
        [
            [0.99, 0, 0],
            [0, 0.49, 0.499],
            [0.99, 1, 1],
        ]
    )
    surv_ensemble = RiskEnsemble(
        [cox_ph, survival_xgboost, lognormal_aft],
        weights,
        eval_time_horizons,
        explainer_plugins=["invase", "kernel_shap"],
        explanations_nepoch=20,
    )

    surv_ensemble.fit(tr_X, tr_T, tr_Y)

    limit = 3
    importance = surv_ensemble.explain(te_X[:limit], eval_time_horizons)

    assert sorted(importance.keys()) == sorted(["invase", "kernel_shap"])
    for src in importance:
        assert importance[src].shape == (limit, te_X.shape[1], len(eval_time_horizons))


def test_risk_estimation_model_compression() -> None:
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    tr_X, te_X, tr_T, te_T, tr_Y, te_Y = train_test_split(
        X, T, Y, test_size=0.1, random_state=0
    )

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    cox_ph = Predictions(category="risk_estimation").get("cox_ph")

    weights = np.asarray(
        [
            [0.99, 0, 0],
            [0, 0.49, 0.499],
        ]
    )
    surv_ensemble = RiskEnsemble(
        [cox_ph, cox_ph, cox_ph],
        weights,
        eval_time_horizons,
    )

    assert len(surv_ensemble.models) == 1
