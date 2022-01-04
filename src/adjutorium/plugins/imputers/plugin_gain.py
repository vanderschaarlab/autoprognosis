# stdlib
from typing import Any, List

# third party
from hyperimpute.plugins.imputers.plugin_gain import plugin as base_model

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.imputers.base as base


class GainPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the GAIN strategy.

    Method:
        Details in the GainImputation class implementation.

    Example:
        >>> import numpy as np
        >>> from adjutorium.plugins.imputers import Imputers
        >>> plugin = Imputers().get("gain")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
    """

    def __init__(self, **kwargs: Any) -> None:
        model = base_model(**kwargs)

        super().__init__(model)

    @staticmethod
    def name() -> str:
        return base_model.name()

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return base_model.hyperparameter_space()


plugin = GainPlugin
