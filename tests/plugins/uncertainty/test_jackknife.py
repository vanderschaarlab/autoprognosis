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
from autoprognosis.plugins.uncertainty.plugin_jackknife import plugin


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("jackknife")]
)
def test_sanity(plugin: Any) -> None:
    uncert_model = plugin(
        Classifiers().get("logistic_regression"),
        n_folds=5,
        random_seed=1,
    )

    assert uncert_model.n_folds == 5
    assert uncert_model.random_seed == 1


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("jackknife")]
)
def test_fit(plugin: Any) -> None:
    uncert_model = plugin(
        Classifiers().get("logistic_regression"),
        n_folds=4,
    )

    X, y = load_breast_cancer(return_X_y=True)

    uncert_model.fit(X, y)

    assert len(uncert_model.models) == 4


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("jackknife")]
)
def test_predict(plugin: Any) -> None:
    uncert_model = plugin(
        Classifiers().get("logistic_regression"),
        n_folds=3,
    )

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    uncert_model.fit(X_train, y_train)

    mean, std = uncert_model.predict(X_test)

    assert mean.size == len(y_test)
    assert std.size == len(y_test)
    assert (mean == y_test).sum() > len(y_test) / 2


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("jackknife")]
)
def test_predict_proba(plugin: Any) -> None:
    uncert_model = plugin(
        Classifiers().get("logistic_regression"),
        n_folds=3,
    )

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    uncert_model.fit(X_train, y_train)

    mean, std = uncert_model.predict_proba(X_test)

    assert mean.shape == (len(y_test), 2)
    assert std.shape == (len(y_test), 2)


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("jackknife")]
)
def test_predict_regressor(plugin: Any) -> None:
    uncert_model = plugin(
        Regression().get("linear_regression"),
    )

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    uncert_model.fit(X_train, y_train)

    mean, confidence = uncert_model.predict(X_test)

    assert len(mean) == len(y_test)
    assert len(confidence) == len(y_test)
    assert confidence.sum() > 0


@pytest.mark.parametrize(
    "plugin", [plugin, UncertaintyQuantification().get_type("jackknife")]
)
def test_predict_survival(plugin: Any) -> None:
    uncert_model = plugin(
        RiskEstimation().get("cox_ph"),
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

    uncert_model.fit(X_train, T_train, y_train)

    mean, confidence = uncert_model.predict(X_test, eval_time_horizons)

    assert mean.shape == (len(y_test), len(eval_time_horizons))
    assert confidence.shape == (len(y_test), len(eval_time_horizons))
