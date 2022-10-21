# stdlib
from abc import ABCMeta, abstractmethod
from typing import Optional

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        import matplotlib.pyplot as plt

        break
    except ImportError:
        depends = ["matplotlib"]
        install(depends)


class ExplainerPlugin(metaclass=ABCMeta):
    def __init__(self, feature_names: list = []) -> None:
        self.feature_names = feature_names

    @staticmethod
    @abstractmethod
    def name() -> str:
        ...

    @staticmethod
    @abstractmethod
    def pretty_name() -> str:
        ...

    @staticmethod
    def type() -> str:
        return "explainer"

    @abstractmethod
    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def plot(
        self,
        importances: pd.DataFrame,
        feature_names: Optional[list] = None,
    ) -> None:

        importances = np.asarray(importances)

        title = f"{self.name()} importance"
        axis_title = "Features"

        if not feature_names:
            feature_names = self.feature_names

        x_pos = np.arange(len(feature_names))

        plt.figure(figsize=(20, 6))
        plt.bar(x_pos, importances, align="center")
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
