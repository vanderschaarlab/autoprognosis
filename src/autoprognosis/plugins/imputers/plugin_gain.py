# stdlib
from typing import Any, List

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.imputers.base as base
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        from hyperimpute.plugins.imputers.plugin_gain import plugin as base_model

        break
    except ImportError:
        depends = ["hyperimpute"]
        install(depends)


class GainPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the GAIN strategy.

    Method:
        Details in the GainImputation class implementation.

    Example:
        >>> import numpy as np
        >>> from autoprognosis.plugins.imputers import Imputers
        >>> plugin = Imputers().get("gain")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
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


plugin = GainPlugin
