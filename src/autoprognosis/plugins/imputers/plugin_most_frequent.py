# stdlib
from typing import Any, List

# third party
from hyperimpute.plugins.imputers.plugin_most_frequent import plugin as base_model

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.imputers.base as base


class MostFrequentPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Most Frequent Imputation strategy.

    Method:
        The Most Frequent Imputation strategy replaces the missing using the most frequent value along each column.

    Example:
        >>> import numpy as np
        >>> from autoprognosis.plugins.imputers import Imputers
        >>> plugin = Imputers().get("most_frequent")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
             0    1    2    3
        0  1.0  1.0  1.0  1.0
        1  1.0  2.0  2.0  1.0
        2  1.0  2.0  2.0  1.0
        3  2.0  2.0  2.0  2.0
    """

    def __init__(self, random_state: int = 0, **kwargs: Any) -> None:
        model = base_model(random_state=random_state, **kwargs)

        super().__init__(model)

    @staticmethod
    def name() -> str:
        return base_model.name()

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return base_model.hyperparameter_space()


plugin = MostFrequentPlugin
