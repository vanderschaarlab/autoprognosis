# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any

# third party
import pandas as pd


class UncertaintyPlugin(metaclass=ABCMeta):
    def __init__(self, model: Any) -> None:
        self.model = model

    @staticmethod
    @abstractmethod
    def name() -> str:
        ...

    @staticmethod
    def type() -> str:
        return "uncertainty_quantification"

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> "UncertaintyPlugin":
        ...

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        ...

    @abstractmethod
    def predict_proba(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        ...
