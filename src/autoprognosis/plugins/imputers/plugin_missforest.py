# stdlib
from typing import Any, List

# third party
from hyperimpute.plugins.imputers.plugin_sklearn_missforest import plugin as base_model

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.imputers.base as base


class MissForestPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the MissForest strategy.

    Method:
        Iterative chained equations(ICE) methods model each feature with missing values as a function of other features in a round-robin fashion. For each step of the round-robin imputation, we use a ExtraTreesRegressor, which fits a number of randomized extra-trees and averages the results.

    Args:
        n_estimators: int, default=10
            The number of trees in the forest.
        max_iter: int, default=500
            maximum number of imputation rounds to perform.
        random_state: int, default set to the current time.
            seed of the pseudo random number generator to use.

    AutoPrognosis Hyperparameters:
        n_estimators: The number of trees in the forest.

    Example:
        >>> import numpy as np
        >>> from autoprognosis.plugins.imputers import Imputers
        >>> plugin = Imputers().get("missforest")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
             0    1    2    3
        0  1.0  1.0  1.0  1.0
        1  1.0  1.9  1.9  1.0
        2  1.0  2.0  2.0  1.0
        3  2.0  2.0  2.0  2.0
    """

    def __init__(self, random_state: int = 0, **kwargs: Any) -> None:
        model = base_model(random_state=random_state, **kwargs)

        super().__init__(model)

    @staticmethod
    def name() -> str:
        return "missforest"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return base_model.hyperparameter_space()


plugin = MissForestPlugin
