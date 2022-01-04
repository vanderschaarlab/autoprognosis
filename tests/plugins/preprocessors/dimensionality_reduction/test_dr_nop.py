# third party
import numpy as np
import pandas as pd
import pytest

# adjutorium absolute
from adjutorium.plugins.preprocessors import PreprocessorPlugin, Preprocessors
from adjutorium.plugins.preprocessors.dimensionality_reduction.plugin_nop import plugin


def from_api() -> PreprocessorPlugin:
    return Preprocessors(category="dimensionality_reduction").get("nop")


def from_module() -> PreprocessorPlugin:
    return plugin()


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nop_plugin_sanity(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nop_plugin_name(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.name() == "nop"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nop_plugin_type(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.type() == "preprocessor"
    assert test_plugin.subtype() == "dimensionality_reduction"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nop_plugin_hyperparams(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.hyperparameter_space() == []


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_nop_plugin_fit_transform(test_plugin: PreprocessorPlugin) -> None:
    res = test_plugin.fit_transform(pd.DataFrame([[1, 1, 1, 1], [2, 2, 2, 2]]))

    np.testing.assert_array_equal(res, [[1, 1, 1, 1], [2, 2, 2, 2]])
