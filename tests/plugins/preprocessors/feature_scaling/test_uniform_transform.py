# third party
import pytest

# adjutorium absolute
from adjutorium.plugins.preprocessors import PreprocessorPlugin, Preprocessors
from adjutorium.plugins.preprocessors.feature_scaling.plugin_uniform_transform import (
    plugin,
)


def from_api() -> PreprocessorPlugin:
    return Preprocessors().get("uniform_transform")


def from_module() -> PreprocessorPlugin:
    return plugin()


def from_serde() -> PreprocessorPlugin:
    buff = plugin().save()
    return plugin().load(buff)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_uniform_transform_plugin_sanity(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_uniform_transform_plugin_name(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.name() == "uniform_transform"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_uniform_transform_plugin_type(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.type() == "preprocessor"
    assert test_plugin.subtype() == "feature_scaling"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_uniform_transform_plugin_hyperparams(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.hyperparameter_space() == []


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_uniform_transform_plugin_fit_transform(
    test_plugin: PreprocessorPlugin,
) -> None:
    res = test_plugin.fit_transform([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])

    assert res.shape == (4, 2)
