# stdlib
from typing import Any, List, Tuple

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.plugins import group
from autoprognosis.plugins.imputers import Imputers
from autoprognosis.plugins.pipeline import Pipeline, PipelineMeta
from autoprognosis.plugins.prediction.classifiers import Classifiers
from autoprognosis.plugins.preprocessors import Preprocessors
from autoprognosis.plugins.utils.simulate import simulate_nan
from autoprognosis.utils.serialization import load_model, save_model


def dataset() -> Tuple:
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2)


def ampute(
    x: np.ndarray, mechanism: str, p_miss: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_simulated = simulate_nan(x, p_miss, mechanism)

    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]

    return x, x_miss, mask


@pytest.mark.parametrize(
    "plugins",
    [
        group(
            [
                Imputers().get_type("ice").fqdn(),
                Classifiers().get_type("perceptron").fqdn(),
            ]
        ),
        group([Classifiers().get_type("perceptron").fqdn()]),
        group(
            [
                Preprocessors().get_type("scaler").fqdn(),
                Classifiers().get_type("perceptron").fqdn(),
            ]
        ),
        group(
            [
                Imputers().get_type("ice").fqdn(),
                Preprocessors().get_type("scaler").fqdn(),
                Classifiers().get_type("perceptron").fqdn(),
            ]
        ),
    ],
)
def test_pipeline_meta_sanity(plugins: Tuple[Any]) -> None:
    dtype = PipelineMeta("meta", plugins, {})

    assert dtype.name() == "->".join(p.name() for p in plugins)
    assert dtype.type() == "->".join(p.type() for p in plugins)
    assert dtype.plugin_types == list(plugins)

    args = {"features_count": 10}
    for act, pl in zip(dtype.hyperparameter_space(**args), plugins):
        assert len(dtype.hyperparameter_space(**args)[act]) == len(
            pl.hyperparameter_space(**args)
        )
        assert len(dtype.hyperparameter_space_for_layer(act, **args)) == len(
            pl.hyperparameter_space(**args)
        )


@pytest.mark.parametrize(
    "plugins",
    [
        group([Imputers().get_type("ice").fqdn()]),
        group([Preprocessors().get_type("scaler").fqdn()]),
        group(
            [
                Classifiers().get_type("perceptron").fqdn(),
                Classifiers().get_type("perceptron").fqdn(),
            ]
        ),
    ],
)
def test_pipeline_meta_invalid(plugins: Tuple[Any]) -> None:
    with pytest.raises(RuntimeError):
        PipelineMeta("meta", plugins, {})()


@pytest.mark.parametrize(
    "plugins_str",
    [
        [
            Imputers().get_type("ice").fqdn(),
            Classifiers().get_type("perceptron").fqdn(),
        ],
        [Classifiers().get_type("perceptron").fqdn()],
        [
            Preprocessors().get_type("scaler").fqdn(),
            Classifiers().get_type("perceptron").fqdn(),
        ],
        [
            Imputers().get_type("ice").fqdn(),
            Preprocessors().get_type("scaler").fqdn(),
            Classifiers().get_type("perceptron").fqdn(),
        ],
    ],
)
def test_pipeline_sanity(plugins_str: List[Any]) -> None:
    dtype = Pipeline(plugins_str)

    plugins = group(plugins_str)

    assert dtype.name() == "->".join(p.name() for p in plugins)
    assert dtype.type() == "->".join(p.type() for p in plugins)

    args = {"features_count": 10}
    for act, pl in zip(dtype.hyperparameter_space(**args), plugins):
        assert len(dtype.hyperparameter_space(**args)[act]) == len(
            pl.hyperparameter_space(**args)
        )


@pytest.mark.parametrize("serialize", [True, False])
def test_pipeline_end2end(serialize: bool) -> None:
    X_train, X_test, y_train, y_test = dataset()
    _, X_train, _ = ampute(X_train, "MAR", 0.1)

    template = Pipeline(
        [
            Imputers().get_type("ice").fqdn(),
            Preprocessors().get_type("minmax_scaler").fqdn(),
            Classifiers().get_type("perceptron").fqdn(),
        ]
    )

    pipeline = template()

    if serialize:
        buff = pipeline.save()
        pipeline = PipelineMeta.load(buff)

    pipeline.fit(pd.DataFrame(X_train), pd.Series(y_train))

    y_pred = pipeline.predict(pd.DataFrame(X_test))

    assert np.abs(np.subtract(y_pred.to_numpy(), y_test)).mean() < 1


def test_pipeline_save_load_template() -> None:
    X_train, X_test, y_train, y_test = dataset()
    X_train[0, 0] = np.nan

    template = Pipeline(
        [
            Imputers().get_type("hyperimpute").fqdn(),
            Preprocessors().get_type("minmax_scaler").fqdn(),
            Classifiers().get_type("neural_nets").fqdn(),
        ]
    )

    params: dict = {}

    pipeline = template(params)

    buff = pipeline.save_template()

    new_pipeline = PipelineMeta.load_template(buff)

    assert pipeline.name() == new_pipeline.name()
    assert pipeline.get_args() == new_pipeline.get_args()


def test_pipeline_save_load() -> None:
    X_train, X_test, y_train, y_test = dataset()
    X_train[0, 0] = np.nan

    template = Pipeline(
        [
            Imputers().get_type("hyperimpute").fqdn(),
            Preprocessors().get_type("minmax_scaler").fqdn(),
            Classifiers().get_type("neural_nets").fqdn(),
        ]
    )

    params: dict = {}

    pipeline = template(params)

    buff = pipeline.save()

    new_pipeline = PipelineMeta.load(buff)

    assert pipeline.name() == new_pipeline.name()
    assert pipeline.get_args() == new_pipeline.get_args()


def test_pipeline_pickle() -> None:
    X_train, X_test, y_train, y_test = dataset()
    X_train[0, 0] = np.nan

    template = Pipeline(
        [
            Imputers().get_type("hyperimpute").fqdn(),
            Preprocessors().get_type("minmax_scaler").fqdn(),
            Classifiers().get_type("neural_nets").fqdn(),
        ]
    )

    params: dict = {}
    pipeline = template(params)

    buff = save_model(pipeline)
    new_pipeline = load_model(buff)

    assert pipeline.name() == new_pipeline.name()
    assert pipeline.get_args() == new_pipeline.get_args()

    pipeline.fit(pd.DataFrame(X_train), pd.Series(y_train))

    buff = save_model(pipeline)
    new_pipeline = load_model(buff)

    assert pipeline.name() == new_pipeline.name()
    assert pipeline.get_args() == new_pipeline.get_args()

    pipeline.predict(pd.DataFrame(X_test))
