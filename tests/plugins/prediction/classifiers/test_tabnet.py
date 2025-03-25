# stdlib
from typing import Any

# third party
import numpy as np
import optuna
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.plugins.prediction import PredictionPlugin, Predictions
from autoprognosis.plugins.prediction.classifiers.plugin_tabnet import plugin
from autoprognosis.utils.serialization import load_model, save_model
from autoprognosis.utils.tester import evaluate_estimator


def from_api() -> PredictionPlugin:
    return Predictions().get("tabnet")


def from_module() -> PredictionPlugin:
    return plugin()


def from_serde() -> PredictionPlugin:
    buff = plugin().save()
    return plugin().load(buff)


def from_pickle() -> PredictionPlugin:
    buff = save_model(plugin())
    return load_model(buff)


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_tabnet_plugin_sanity(test_plugin: PredictionPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_tabnet_plugin_name(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.name() == "tabnet"


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_tabnet_plugin_type(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.type() == "prediction"
    assert test_plugin.subtype() == "classifier"


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_tabnet_plugin_hyperparams(test_plugin: PredictionPlugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 8


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_tabnet_plugin_fit_predict(test_plugin: PredictionPlugin) -> None:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    y_pred = test_plugin.fit(X_train, y_train).predict(X_test).to_numpy()

    assert np.abs(np.subtract(y_pred, y_test)).mean() < 2


@pytest.mark.slow
def test_param_search() -> None:
    if len(plugin.hyperparameter_space()) == 0:
        return

    expected_len = 10

    X, y = load_iris(return_X_y=True)

    def evaluate_args(**kwargs: Any) -> float:
        kwargs["n_layers_hidden"] = 1
        kwargs["n_units_hidden"] = 2
        model = plugin(**kwargs)
        metrics = evaluate_estimator(model, X, y)

        return metrics["raw"]["aucroc"][0]

    def objective(trial: optuna.Trial) -> float:
        args = plugin.sample_hyperparameters(trial)
        return evaluate_args(**args)

    study = optuna.create_study(
        load_if_exists=True,
        directions=["maximize"],
        study_name=f"test_param_search_{plugin.name()}",
    )
    study.optimize(objective, n_trials=expected_len, timeout=60)

    FAILURE_TOL = 0.20
    expect_above = int(round(expected_len * FAILURE_TOL))
    assert expected_len - len(study.trials) <= expect_above
