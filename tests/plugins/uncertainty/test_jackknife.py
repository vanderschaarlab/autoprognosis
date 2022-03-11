# stdlib
from typing import Any

# third party
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# adjutorium absolute
from adjutorium.plugins.prediction.classifiers import Classifiers
from adjutorium.plugins.uncertainty import UncertaintyQuantification
from adjutorium.plugins.uncertainty.plugin_jackknife import plugin


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
