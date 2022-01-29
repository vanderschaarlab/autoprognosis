# stdlib
from typing import Any, List

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.imputers.base as base
from adjutorium.utils.pip import install

for retry in range(2):
    try:
        # third party
        from hyperimpute.plugins.imputers.plugin_sinkhorn import plugin as base_model

        break
    except ImportError:
        depends = ["hyperimpute"]
        install(depends)


class SinkhornPlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Sinkhorn strategy.

    Method:
        Details in the SinkhornImputation class implementation.

    Example:
        >>> import numpy as np
        >>> from adjutorium.plugins.imputers import Imputers
        >>> plugin = Imputers().get("sinkhorn")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
                  0         1         2         3
        0  1.000000  1.000000  1.000000  1.000000
        1  1.404637  1.651113  1.651093  1.404638
        2  1.000000  2.000000  2.000000  1.000000
        3  2.000000  2.000000  2.000000  2.000000
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


plugin = SinkhornPlugin
