# third party
import pytest
from lifelines.datasets import load_rossi
from sklearn.datasets import load_diabetes, load_iris

# autoprognosis absolute
from autoprognosis.plugins.prediction import Predictions
from autoprognosis.utils.tester import (
    evaluate_estimator,
    evaluate_estimator_multiple_seeds,
    evaluate_regression,
    evaluate_regression_multiple_seeds,
    evaluate_survival_estimator,
    evaluate_survival_estimator_multiple_seeds,
)

clf_supported_metrics = [
    "aucroc",
    "aucprc",
    "accuracy",
    "f1_score_micro",
    "f1_score_macro",
    "f1_score_weighted",
    "kappa",
    "precision_micro",
    "precision_macro",
    "precision_weighted",
    "recall_micro",
    "recall_macro",
    "recall_weighted",
    "mcc",
]
reg_supported_metrics = [
    "r2",
    "mse",
    "mae",
]
surv_supported_metrics = [
    "c_index",
    "brier_score",
    "aucroc",
    "sensitivity",
    "specificity",
    "PPV",
    "NPV",
    "predicted_cases",
]


@pytest.mark.parametrize("n_folds", [2, 5])
def test_classifier(n_folds: int) -> None:
    model = Predictions().get("logistic_regression")
    X, y = load_iris(return_X_y=True)

    metrics = evaluate_estimator(model, X, y, n_folds=n_folds)

    for metric in clf_supported_metrics:
        assert metric in metrics["raw"]
        assert metric in metrics["str"]


def test_classifier_multiple_seeds() -> None:
    model = Predictions().get("logistic_regression")
    X, y = load_iris(return_X_y=True)

    metrics = evaluate_estimator_multiple_seeds(model, X, y, seeds=[0, 1])

    for seed in [0, 1]:
        for metric in clf_supported_metrics:
            assert metric in metrics["seeds"][seed]
            assert metric in metrics["str"]


@pytest.mark.parametrize("n_folds", [2, 5])
def test_reg(n_folds: int) -> None:
    model = Predictions(category="regression").get("linear_regression")
    X, y = load_diabetes(return_X_y=True)

    metrics = evaluate_regression(model, X, y, n_folds=n_folds)

    for metric in reg_supported_metrics:
        assert metric in metrics["raw"]
        assert metric in metrics["str"]


def test_reg_multiple_seeds() -> None:
    model = Predictions(category="regression").get("linear_regression")
    X, y = load_diabetes(return_X_y=True)

    metrics = evaluate_regression_multiple_seeds(model, X, y, seeds=[0, 1])

    for seed in [0, 1]:
        for metric in reg_supported_metrics:
            assert metric in metrics["seeds"][seed]
            assert metric in metrics["str"]


@pytest.mark.parametrize("n_folds", [2, 5])
def test_surv(n_folds: int) -> None:
    model = Predictions(category="risk_estimation").get("cox_ph")
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.25)),
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    metrics = evaluate_survival_estimator(
        model, X, T, Y, time_horizons=eval_time_horizons, n_folds=n_folds
    )

    for metric in surv_supported_metrics:
        assert metric in metrics["raw"]
        assert metric in metrics["str"]


def test_surv_multiple_seeds() -> None:
    model = Predictions(category="risk_estimation").get("cox_ph")
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
    ]

    metrics = evaluate_survival_estimator_multiple_seeds(
        model, X, T, Y, time_horizons=eval_time_horizons, seeds=[0, 1]
    )

    for seed in [0, 1]:
        for metric in surv_supported_metrics:
            assert metric in metrics["seeds"][seed]
            assert metric in metrics["str"]
