# stdlib
from typing import Any

# third party
import optuna
import pytest
from sklearn.datasets import load_diabetes

# autoprognosis absolute
from autoprognosis.plugins.prediction import PredictionPlugin, Predictions
from autoprognosis.plugins.prediction.regression.plugin_tabnet_regressor import plugin
from autoprognosis.utils.serialization import load_model, save_model
from autoprognosis.utils.tester import evaluate_regression

n_iter = 500


def from_api() -> PredictionPlugin:
    return Predictions(category="regression").get("tabnet_regressor", n_iter=n_iter)


def from_module() -> PredictionPlugin:
    return plugin(n_iter=n_iter)


def from_serde() -> PredictionPlugin:
    buff = plugin(n_iter=n_iter).save()
    return plugin().load(buff)


def from_pickle() -> PredictionPlugin:
    buff = save_model(plugin(n_iter=n_iter))
    return load_model(buff)


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_tabnet_regressor_plugin_sanity(test_plugin: PredictionPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_tabnet_regressor_plugin_name(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.name() == "tabnet_regressor"


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_tabnet_regressor_plugin_type(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.type() == "prediction"
    assert test_plugin.subtype() == "regression"


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_tabnet_regressor_plugin_hyperparams(
    test_plugin: PredictionPlugin,
) -> None:
    assert len(test_plugin.hyperparameter_space()) == 8


@pytest.mark.slow
@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_tabnet_regressor_plugin_fit_predict(
    test_plugin: PredictionPlugin,
) -> None:
    X, y = load_diabetes(return_X_y=True)

    score = evaluate_regression(test_plugin, X, y)

    assert score["raw"]["mse"][0] < 5100


@pytest.mark.slow
def test_param_search() -> None:
    if len(plugin.hyperparameter_space()) == 0:
        return

    expected_len = 10

    X, y = load_diabetes(return_X_y=True)

    def evaluate_args(**kwargs: Any) -> float:
        kwargs["n_iter"] = expected_len

        model = plugin(**kwargs)
        metrics = evaluate_regression(model, X, y)

        return metrics["raw"]["mse"][0]

    def objective(trial: optuna.Trial) -> float:
        args = plugin.sample_hyperparameters(trial)
        return evaluate_args(**args)

    study = optuna.create_study(
        load_if_exists=True,
        directions=["maximize"],
        study_name=f"test_param_search_{plugin.name()}",
    )
    study.optimize(objective, n_trials=10, timeout=60)

    FAILURE_TOL = 0.20
    expect_above = int(round(expected_len * FAILURE_TOL))
    assert expected_len - len(study.trials) <= expect_above
