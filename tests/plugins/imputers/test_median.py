# third party
import numpy as np
import pandas as pd
import pytest

# autoprognosis absolute
from autoprognosis.plugins.imputers import ImputerPlugin, Imputers
from autoprognosis.plugins.imputers.plugin_median import plugin


def from_api() -> ImputerPlugin:
    return Imputers().get("median")


def from_module() -> ImputerPlugin:
    return plugin()


def from_serde() -> ImputerPlugin:
    buff = plugin().save()
    return plugin().load(buff)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_median_plugin_sanity(test_plugin: ImputerPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_median_plugin_name(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.name() == "median"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_median_plugin_type(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.type() == "imputer"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_median_plugin_hyperparams(test_plugin: ImputerPlugin) -> None:
    assert test_plugin.hyperparameter_space() == []


@pytest.mark.parametrize("test_plugin", [from_api(), from_module(), from_serde()])
def test_median_plugin_fit_transform(test_plugin: ImputerPlugin) -> None:
    res = test_plugin.fit_transform(
        pd.DataFrame(
            [[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [3, 3, 9, 9], [2, 2, 2, 2]]
        )
    )

    np.testing.assert_array_equal(
        res, [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 9, 9], [2, 2, 2, 2]]
    )
