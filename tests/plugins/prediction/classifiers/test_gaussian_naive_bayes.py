# stdlib
from typing import Any

# third party
import numpy as np
import optuna
import pytest

# autoprognosis absolute
from autoprognosis.plugins.prediction import PredictionPlugin, Predictions
from autoprognosis.plugins.prediction.classifiers.plugin_gaussian_naive_bayes import (
    plugin,
)
from autoprognosis.utils.serialization import load_model, save_model
from autoprognosis.utils.tester import evaluate_estimator


def from_api() -> PredictionPlugin:
    return Predictions().get("gaussian_naive_bayes")


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
def test_gaussian_naive_bayes_plugin_sanity(test_plugin: PredictionPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_gaussian_naive_bayes_plugin_name(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.name() == "gaussian_naive_bayes"


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_gaussian_naive_bayes_plugin_type(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.type() == "prediction"
    assert test_plugin.subtype() == "classifier"


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_gaussian_naive_bayes_plugin_hyperparams(
    test_plugin: PredictionPlugin,
) -> None:
    assert len(test_plugin.hyperparameter_space()) == 0


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_gaussian_naive_bayes_plugin_fit_predict(
    test_plugin: PredictionPlugin,
) -> None:
    rng = np.random.RandomState(1)

    X = rng.randint(6, size=(6, 100))
    y = [1, 2, 3, 4, 4, 5]

    y_pred = test_plugin.fit(X, y).predict(X).to_numpy()

    assert (y_pred == [[1], [2], [3], [4], [4], [5]]).all()


def test_param_search() -> None:
    if len(plugin.hyperparameter_space()) == 0:
        return

    rng = np.random.RandomState(1)

    N = 1000
    X = rng.randint(N, size=(N, 3))
    y = rng.randint(2, size=(N))

    def evaluate_args(**kwargs: Any) -> float:
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
    study.optimize(objective, n_trials=10, timeout=60)

    assert len(study.trials) == 10
