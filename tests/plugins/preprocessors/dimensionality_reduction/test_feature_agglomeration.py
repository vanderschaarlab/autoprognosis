# third party
import pytest
from sklearn.datasets import load_iris

# autoprognosis absolute
from autoprognosis.plugins.preprocessors import PreprocessorPlugin, Preprocessors
from autoprognosis.plugins.preprocessors.dimensionality_reduction.plugin_feature_agglomeration import (
    plugin,
)

n_clusters = 2


def from_api() -> PreprocessorPlugin:
    return Preprocessors(category="dimensionality_reduction").get(
        "feature_agglomeration", n_clusters=n_clusters
    )


def from_module() -> PreprocessorPlugin:
    return plugin(n_clusters=n_clusters)


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_feature_agglomeration_plugin_sanity(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin is not None


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_feature_agglomeration_plugin_name(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.name() == "feature_agglomeration"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_feature_agglomeration_plugin_type(test_plugin: PreprocessorPlugin) -> None:
    assert test_plugin.type() == "preprocessor"
    assert test_plugin.subtype() == "dimensionality_reduction"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_feature_agglomeration_plugin_hyperparams(
    test_plugin: PreprocessorPlugin,
) -> None:
    kwargs = {"features_count": 2}
    assert len(test_plugin.hyperparameter_space(**kwargs)) == 1
    assert test_plugin.hyperparameter_space(**kwargs)[0].name == "n_clusters"


@pytest.mark.parametrize("test_plugin", [from_api(), from_module()])
def test_feature_agglomeration_plugin_fit_transform(
    test_plugin: PreprocessorPlugin,
) -> None:
    X, y = load_iris(return_X_y=True)
    res = test_plugin.fit_transform(X, y)

    assert res.shape == (len(X), n_clusters)
