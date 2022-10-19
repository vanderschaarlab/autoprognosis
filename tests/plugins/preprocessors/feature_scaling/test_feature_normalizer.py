# third party
import numpy as np
import pytest

# autoprognosis absolute
from autoprognosis.plugins.preprocessors import PreprocessorPlugin, Preprocessors
from autoprognosis.plugins.preprocessors.feature_scaling.plugin_feature_normalizer import (
    plugin,
)
from autoprognosis.utils.serialization import load_model, save_model


def from_api() -> PreprocessorPlugin:
    return Preprocessors().get("feature_normalizer")


def from_module() -> PreprocessorPlugin:
    return plugin()


def from_serde() -> PreprocessorPlugin:
    buff = plugin().save()
    return plugin().load(buff)


def from_pickle() -> PreprocessorPlugin:
    buff = save_model(plugin())
    return load_model(buff)


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_feature_normalizer_plugin_sanity(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_feature_normalizer_plugin_name(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.name() == "feature_normalizer"


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_feature_normalizer_plugin_type(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.type() == "preprocessor"
    assert test_plugin.subtype() == "feature_scaling"


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_feature_normalizer_plugin_hyperparams(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.hyperparameter_space() == []


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_feature_normalizer_plugin_fit_transform(
    test_plugin: PreprocessorPlugin,
) -> None:
    res = test_plugin.fit_transform([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])

    np.testing.assert_array_equal(
        res, [[0.8, 0.2, 0.4, 0.4], [0.1, 0.3, 0.9, 0.3], [0.5, 0.7, 0.5, 0.1]]
    )
