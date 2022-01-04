# stdlib
from typing import Any, List

# third party
import pandas as pd

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.preprocessors.base as base


class NopPlugin(base.PreprocessorPlugin):
    """Preprocessing plugin that doesn't alter the dataset."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def name() -> str:
        return "nop"

    @staticmethod
    def subtype() -> str:
        return "feature_scaling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "NopPlugin":
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X

    def save(self) -> bytes:
        return b""

    @classmethod
    def load(cls, buff: bytes) -> "NopPlugin":
        return cls()


plugin = NopPlugin
