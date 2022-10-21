# third party
import pytest
from sklearn.datasets import load_iris

# autoprognosis absolute
from autoprognosis.plugins.preprocessors import PreprocessorPlugin, Preprocessors
from autoprognosis.plugins.preprocessors.dimensionality_reduction.plugin_fast_ica import (
    plugin,
)

n_components = 3


def from_api() -> PreprocessorPlugin:
    return Preprocessors(category="dimensionality_reduction").get(
        "fast_ica", n_components=n_components
    )


def from_module() -> PreprocessorPlugin:
    return plugin(n_components=n_components)


def from_serde() -> PreprocessorPlugin:
    buff = plugin(n_components=n_components).save()
    return plugin().load(buff)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_fast_ica_plugin_sanity(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_fast_ica_plugin_name(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.name() == "fast_ica"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_fast_ica_plugin_type(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.type() == "preprocessor"
    assert test_plugin.subtype() == "dimensionality_reduction"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_fast_ica_plugin_hyperparams(test_plugin: PreprocessorPlugin) -> None:
    kwargs = {"features_count": 2}
    assert len(test_plugin.hyperparameter_space(**kwargs)) == 1
    assert test_plugin.hyperparameter_space(**kwargs)[0].name == "n_components"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_fast_ica_plugin_fit_transform(test_plugin: PreprocessorPlugin) -> None:
    X, y = load_iris(return_X_y=True)
    res = test_plugin.fit_transform(X, y)

    assert res.shape == (len(X), n_components)
