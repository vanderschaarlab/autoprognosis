# third party
import pytest

# autoprognosis absolute
from autoprognosis.plugins.preprocessors import PreprocessorPlugin, Preprocessors
from autoprognosis.plugins.preprocessors.dimensionality_reduction.plugin_data_cleanup import (
    plugin,
)
from autoprognosis.utils.serialization import load_model, save_model


def from_api() -> PreprocessorPlugin:
    return Preprocessors(category="dimensionality_reduction").get("data_cleanup")


def from_module() -> PreprocessorPlugin:
    return plugin()


def from_serde() -> PreprocessorPlugin:
    buff = plugin().save()
    return plugin.load(buff)


def from_pickle() -> PreprocessorPlugin:
    buff = save_model(plugin())
    return load_model(buff)


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_variance_threshold_plugin_sanity(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_variance_threshold_plugin_name(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.name() == "data_cleanup"


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_variance_threshold_plugin_type(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.type() == "preprocessor"
    assert test_plugin.subtype() == "dimensionality_reduction"


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_variance_threshold_plugin_hyperparams(
    test_plugin: PreprocessorPlugin,
) -> None:
    assert test_plugin.hyperparameter_space() == []


@pytest.mark.parametrize(
    "test_plugin", [from_api(), from_module(), from_serde(), from_pickle()]
)
def test_variance_threshold_plugin_fit_transform(
    test_plugin: PreprocessorPlugin,
) -> None:
    res = test_plugin.fit_transform(
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 9, 9], [2, 2, 2, 2]], [1, 2, 3, 4]
    )

    assert res.shape == (4, 2)
