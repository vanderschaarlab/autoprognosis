# stdlib
from typing import Any, List

# third party
from hyperimpute.plugins.imputers.plugin_softimpute import plugin as base_model

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.imputers.base as base


class SoftImputePlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the SoftImpute strategy.

    Method:
        Details in the SoftImpute class implementation.

    Example:
        >>> import numpy as np
        >>> from adjutorium.plugins.imputers import Imputers
        >>> plugin = Imputers().get("softimpute")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
                      0             1             2             3
        0  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00
        1  3.820605e-16  1.708249e-16  1.708249e-16  3.820605e-16
        2  1.000000e+00  2.000000e+00  2.000000e+00  1.000000e+00
        3  2.000000e+00  2.000000e+00  2.000000e+00  2.000000e+00
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


plugin = SoftImputePlugin
