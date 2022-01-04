# third party
import pytest

# adjutorium absolute
from adjutorium.plugins.preprocessors import PreprocessorPlugin, Preprocessors
from adjutorium.plugins.preprocessors.dimensionality_reduction.plugin_gauss_projection import (
    plugin,
)

n_components = 3


def from_api() -> PreprocessorPlugin:
    return Preprocessors(category="dimensionality_reduction").get(
        "gauss_projection", n_components=n_components
    )


def from_module() -> PreprocessorPlugin:
    return plugin(n_components=n_components)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_gauss_projection_plugin_sanity(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_gauss_projection_plugin_name(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.name() == "gauss_projection"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_gauss_projection_plugin_type(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.type() == "preprocessor"
    assert test_plugin.subtype() == "dimensionality_reduction"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_gauss_projection_plugin_hyperparams(test_plugin: PreprocessorPlugin) -> None:
    kwargs = {"features_count": 2}
    assert len(test_plugin.hyperparameter_space(**kwargs)) == 1
    assert test_plugin.hyperparameter_space(**kwargs)[0].name == "n_components"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_gauss_projection_plugin_fit_transform(test_plugin: PreprocessorPlugin) -> None:
    res = test_plugin.fit_transform(
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 9, 9], [2, 2, 2, 2]]
    )

    assert res.shape == (4, n_components)
