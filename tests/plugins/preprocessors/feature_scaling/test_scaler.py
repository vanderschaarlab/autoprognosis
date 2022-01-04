# third party
import numpy as np
import pytest

# adjutorium absolute
from adjutorium.plugins.preprocessors import PreprocessorPlugin, Preprocessors
from adjutorium.plugins.preprocessors.feature_scaling.plugin_scaler import plugin


def from_api() -> PreprocessorPlugin:
    return Preprocessors().get("scaler")


def from_module() -> PreprocessorPlugin:
    return plugin()


def from_serde() -> PreprocessorPlugin:
    buff = plugin().save()
    return plugin().load(buff)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_scaler_plugin_sanity(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_scaler_plugin_name(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.name() == "scaler"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_scaler_plugin_type(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.type() == "preprocessor"
    assert test_plugin.subtype() == "feature_scaling"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_scaler_plugin_hyperparams(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.hyperparameter_space() == []


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_scaler_plugin_fit_transform(test_plugin: PreprocessorPlugin) -> None:
    res = test_plugin.fit_transform([[0, 0], [0, 0], [1, 1], [1, 1]])

    np.testing.assert_array_equal(
        res, [[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]]
    )
