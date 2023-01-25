# third party
from sklearn.datasets import load_diabetes, load_iris

# autoprognosis absolute
from autoprognosis.plugins.prediction import Predictions
from autoprognosis.utils.tester import (
    evaluate_estimator,
    evaluate_estimator_multiple_seeds,
    evaluate_regression,
    evaluate_regression_multiple_seeds,
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
    "rmse",
]


def test_classifier() -> None:
    model = Predictions().get("logistic_regression")
    X, y = load_iris(return_X_y=True)

    metrics = evaluate_estimator(model, X, y)

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


def test_reg() -> None:
    model = Predictions(category="regression").get("linear_regression")
    X, y = load_diabetes(return_X_y=True)

    metrics = evaluate_regression(model, X, y)

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
