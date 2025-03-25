# third party
import pytest
from pycox.datasets import metabric
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.plugins.prediction import PredictionPlugin, Predictions
from autoprognosis.plugins.prediction.risk_estimation.plugin_deephit import plugin
from autoprognosis.utils.metrics import evaluate_brier_score, evaluate_c_index


def from_api() -> PredictionPlugin:
    return Predictions(category="risk_estimation").get(
        "deephit",
    )


def from_module() -> PredictionPlugin:
    return plugin()


def from_serde() -> PredictionPlugin:
    buff = plugin().save()
    return plugin().load(buff)


def calibrated(method: int) -> PredictionPlugin:
    return plugin(calibration=method)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_deephit_plugin_sanity(test_plugin: PredictionPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_deephit_plugin_name(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.name() == "deephit"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_deephit_plugin_type(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.type() == "prediction"
    assert test_plugin.subtype() == "risk_estimation"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_deephit_plugin_hyperparams(test_plugin: PredictionPlugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 7


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        from_module(),
    ],
)
def test_deephit_plugin_fit_predict(test_plugin: PredictionPlugin) -> None:
    df = metabric.read_df()

    X = df.drop(["duration", "event"], axis=1)
    Y = df["event"]
    T = df["duration"]

    X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
        X, T, Y, test_size=0.1, random_state=0
    )

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    y_pred = (
        test_plugin.fit(X_train, T_train, Y_train, eval_times=eval_time_horizons)
        .predict(X_test, T_test)
        .to_numpy()
    )

    for e_idx, eval_time in enumerate(eval_time_horizons):
        c_index = evaluate_c_index(
            T_train, Y_train, y_pred[:, e_idx], T_test, Y_test, eval_time
        )
        assert c_index > 0.5

        brier_score = evaluate_brier_score(
            T_train, Y_train, y_pred[:, e_idx], T_test, Y_test, eval_time
        )
        assert brier_score < 0.5
