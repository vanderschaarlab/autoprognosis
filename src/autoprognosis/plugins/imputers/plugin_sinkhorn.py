# stdlib
from typing import Any, List

# third party
from hyperimpute.plugins.imputers.plugin_sinkhorn import plugin as base_model

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.imputers.base as base


class SinkhornPlugin(base.ImputerPlugin):
    """Sinkhorn imputation can be used to impute quantitative data and it relies on the idea that two batches        extracted randomly from the same dataset should share the same distribution and consists in minimizing optimal       transport distances between batches.

     Args:
         eps: float, default=0.01
             Sinkhorn regularization parameter.
         lr : float, default = 0.01
             Learning rate.
         opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
             Optimizer class to use for fitting.
         n_epochs : int, default=15
             Number of gradient updates for each model within a cycle.
         batch_size : int, defatul=256
             Size of the batches on which the sinkhorn divergence is evaluated.
         n_pairs : int, default=10
             Number of batch pairs used per gradient update.
         noise : float, default = 0.1
             Noise used for the missing values initialization.
         scaling: float, default=0.9
             Scaling parameter in Sinkhorn iterations

    Example:
        >>> import numpy as np
        >>> from autoprognosis.plugins.imputers import Imputers
        >>> plugin = Imputers().get("sinkhorn")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
                  0         1         2         3
        0  1.000000  1.000000  1.000000  1.000000
        1  1.404637  1.651113  1.651093  1.404638
        2  1.000000  2.000000  2.000000  1.000000
        3  2.000000  2.000000  2.000000  2.000000

    Reference: "Missing Data Imputation using Optimal Transport", Boris Muzellec, Julie Josse, Claire Boyer, Marco   Cuturi
     Original code: https://github.com/BorisMuzellec/MissingDataOT
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


plugin = SinkhornPlugin
