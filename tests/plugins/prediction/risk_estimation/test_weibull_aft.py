# third party
import pytest
from lifelines.datasets import load_rossi
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.plugins.prediction import PredictionPlugin, Predictions
from autoprognosis.plugins.prediction.risk_estimation.plugin_weibull_aft import plugin
from autoprognosis.utils.metrics import evaluate_brier_score, evaluate_c_index


def from_api() -> PredictionPlugin:
    return Predictions(category="risk_estimation").get(
        "weibull_aft",
        with_explanations=True,
        explanations_nepoch=100,
        explanations_nfolds=1,
    )


def from_module() -> PredictionPlugin:
    return plugin(
        with_explanations=True, explanations_nepoch=200, explanations_nfolds=1
    )


def from_serde() -> PredictionPlugin:
    buff = plugin().save()
    return plugin().load(buff)


def calibrated(method: int) -> PredictionPlugin:
    return plugin(calibration=method)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_weibull_aft_plugin_sanity(test_plugin: PredictionPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_weibull_aft_plugin_name(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.name() == "weibull_aft"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_weibull_aft_plugin_type(test_plugin: PredictionPlugin) -> None:
    assert test_plugin.type() == "prediction"
    assert test_plugin.subtype() == "risk_estimation"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_weibull_aft_plugin_hyperparams(test_plugin: PredictionPlugin) -> None:
    assert len(test_plugin.hyperparameter_space()) == 2
    assert test_plugin.hyperparameter_space()[0].name == "alpha"
    assert test_plugin.hyperparameter_space()[1].name == "l1_ratio"


@pytest.mark.parametrize(
    "test_plugin",
    [
        from_api(),
        from_module(),
    ],
)
def test_weibull_aft_plugin_fit_predict(test_plugin: PredictionPlugin) -> None:
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

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

    importance = test_plugin.explain(X_train)
    assert importance.shape == (
        X_train.shape[0],
        X_train.shape[1],
        len(eval_time_horizons),
    )
