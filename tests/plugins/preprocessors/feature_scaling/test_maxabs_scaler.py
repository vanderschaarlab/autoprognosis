# third party
import numpy as np
import pytest

# autoprognosis absolute
from autoprognosis.plugins.preprocessors import PreprocessorPlugin, Preprocessors
from autoprognosis.plugins.preprocessors.feature_scaling.plugin_maxabs_scaler import (
    plugin,
)


def from_api() -> PreprocessorPlugin:
    return Preprocessors().get("maxabs_scaler")


def from_module() -> PreprocessorPlugin:
    return plugin()


def from_serde() -> PreprocessorPlugin:
    buff = plugin().save()
    return plugin().load(buff)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_maxabs_scaler_plugin_sanity(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_maxabs_scaler_plugin_name(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.name() == "maxabs_scaler"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_maxabs_scaler_plugin_type(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.type() == "preprocessor"
    assert test_plugin.subtype() == "feature_scaling"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_maxabs_scaler_plugin_hyperparams(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.hyperparameter_space() == []


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_maxabs_scaler_plugin_fit_transform(test_plugin: PreprocessorPlugin) -> None:
    res = test_plugin.fit_transform(
        [[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]]
    )

    np.testing.assert_array_equal(
        res, [[0.5, -1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, -0.5]]
    )
