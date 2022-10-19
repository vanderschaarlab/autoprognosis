# stdlib
from typing import Any

# third party
import pandas as pd

# autoprognosis absolute
import autoprognosis.plugins.core.base_plugin as plugin
import autoprognosis.plugins.utils.decorators as decorators
from autoprognosis.utils.serialization import load_model, save_model


class ImputerPlugin(plugin.Plugin):
    """Base class for the imputation plugins.

    It provides the implementation for plugin.Plugin.type() static method.

    Each derived class must implement the following methods(inherited from plugin.Plugin):
        name() - a static method that returns the name of the plugin. e.g., EM, mice, etc.
        hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `Params` derived objects.
        _fit() - internal implementation, called by the `fit()` method.
        _transform() - internal implementation, called by the `transform()` method.

    If any method implementation is missing, the class constructor will fail.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()

        if not hasattr(model, "fit") or not hasattr(model, "transform"):
            raise RuntimeError("Invalid instance model type")

        self._model = model

    @staticmethod
    def type() -> str:
        return "imputer"

    @staticmethod
    def subtype() -> str:
        return "default"

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError(
            "Imputation plugins do not implement the 'predict' method"
        )

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
            "Imputation plugins do not implement the 'predict_proba' method"
        )

    @decorators.benchmark
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "ImputerPlugin":
        return self._model.fit(X, *args, **kwargs)

    @decorators.benchmark
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.transform(X)

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "ImputerPlugin":
        obj = load_model(buff)

        if not isinstance(obj, cls):
            raise RuntimeError("Invalid object type in buffer")

        return obj
