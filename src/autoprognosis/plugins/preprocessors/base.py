# stdlib
from typing import Any, Tuple

# third party
import pandas as pd

# autoprognosis absolute
import autoprognosis.plugins.core.base_plugin as plugin


class PreprocessorPlugin(plugin.Plugin):
    """Base class for the preprocessing plugins.

    It provides the implementation for plugin.Plugin.type() static method.

    Each derived class must implement the following methods(inherited from plugin.Plugin):
        name() - a static method that returns the name of the plugin.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `params.Params` derived objects.
        _fit() - internal implementation, called by the `fit` method.
        _transform() - internal implementation, called by the `transform` method.

    If any method implementation is missing, the class constructor will fail.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def type() -> str:
        return "preprocessor"

    @staticmethod
    def components_interval(*args: Any, **kwargs: Any) -> Tuple[int, int]:
        if "features_count" not in kwargs:
            raise ValueError(
                "invalid arguments for hyperparameter_space. Expecting 'features_count' value"
            )

        feature_count = kwargs.get("features_count", 0)

        if feature_count == 0:
            raise ValueError("invalid value for 'features_count'")

        return (1, feature_count)

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError(
            "Preprocessing plugins do not implement the 'predict' method"
        )

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "Preprocessing plugins do not implement the 'predict_proba' method"
        )
