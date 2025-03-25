# stdlib
from typing import Any

import pytest

# third party
from lifelines.datasets import load_rossi
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.plugins.prediction.classifiers import Classifiers
from autoprognosis.plugins.prediction.regression import Regression
from autoprognosis.plugins.prediction.risk_estimation import RiskEstimation
from autoprognosis.plugins.uncertainty import UncertaintyQuantification
from autoprognosis.plugins.uncertainty.plugin_conformal_prediction import plugin


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("conformal_prediction")]
)
def test_sanity(plugin: Any) -> None:
    uncert_model = plugin(
        Classifiers().get("logistic_regression"),
        task_type="classification",
        random_seed=1,
    )

    assert uncert_model.random_seed == 1
    assert uncert_model.task_type == "classification"


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("conformal_prediction")]
)
def test_fit(plugin: Any) -> None:
    uncert_model = plugin(
        Classifiers().get("logistic_regression"),
    )

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    uncert_model.fit(X, y)

    assert len(uncert_model.calibration) == 2


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("conformal_prediction")]
)
def test_predict_classifier(plugin: Any) -> None:
    uncert_model = plugin(
        Classifiers().get("logistic_regression"),
    )

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    uncert_model.fit(X_train, y_train)

    mean, confidence = uncert_model.predict(X_test)

    assert len(mean) == len(y_test)
    assert len(confidence) == len(y_test)
    assert (mean.values == y_test.values).sum() > len(y_test) / 2
    assert confidence[mean.values != y_test.values].max() < 0.9


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("conformal_prediction")]
)
def test_predict_proba_classifier(plugin: Any) -> None:
    uncert_model = plugin(
        Classifiers().get("logistic_regression"),
    )

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    uncert_model.fit(X_train, y_train)

    mean, confidence = uncert_model.predict_proba(X_test)

    assert len(mean) == len(y_test)
    assert len(confidence) == len(y_test)


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("conformal_prediction")]
)
def test_predict_regressor(plugin: Any) -> None:
    uncert_model = plugin(
        Regression().get("linear_regression"),
        task_type="regression",
    )

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    uncert_model.fit(X_train, y_train)

    mean, confidence = uncert_model.predict(X_test)

    assert len(mean) == len(y_test)
    assert len(confidence) == len(y_test)
    assert confidence.sum() > 0


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("conformal_prediction")]
)
def test_predict_survival(plugin: Any) -> None:
    uncert_model = plugin(
        RiskEstimation().get("cox_ph"),
        task_type="risk_estimation",
    )
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]
    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    X_train, X_test, T_train, T_test, y_train, y_test = train_test_split(X, T, Y)

    uncert_model.fit(X_train, T_train, y_train, time_horizons=eval_time_horizons)

    mean, confidence = uncert_model.predict(X_test, time_horizons=eval_time_horizons)

    assert mean.shape == (len(y_test), len(eval_time_horizons))
    assert confidence.shape == (len(y_test), len(eval_time_horizons))
