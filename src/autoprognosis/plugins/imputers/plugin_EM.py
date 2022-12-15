# stdlib
from typing import Any, List

# third party
from hyperimpute.plugins.imputers.plugin_EM import plugin as base_model

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.imputers.base as base


class EMPlugin(base.ImputerPlugin):
    """The EM algorithm is an optimization algorithm that assumes a distribution for the partially missing data and  tries to maximize the expected complete data log-likelihood under that distribution.

     Steps:
         1. For an input dataset X with missing values, we assume that the values are sampled from distribution       N(Mu, Sigma).
         2. We generate the "observed" and "missing" masks from X, and choose some initial values for Mu = Mu0 and    Sigma = Sigma0.
         3. The EM loop tries to approximate the (Mu, Sigma) pair by some iterative means under the conditional       distribution of missing components.
         4. The E step finds the conditional expectation of the "missing" data, given the observed values and         current estimates of the parameters. These expectations are then substituted for the "missing" data.
         5. In the M step, maximum likelihood estimates of the parameters are computed as though the missing data     had been filled in.
         6. The X_reconstructed contains the approximation after each iteration.

     Args:
         maxit: int, default=500
             maximum number of imputation rounds to perform.
         convergence_threshold : float, default=1e-08
             Minimum ration difference between iterations before stopping.
        random_state: int
            Random seed

     Paper: "Maximum Likelihood from Incomplete Data via the EM Algorithm", A. P. Dempster, N. M. Laird and D. B.    Rubin

    Example:
        >>> import numpy as np
        >>> from autoprognosis.plugins.imputers import Imputers
        >>> plugin = Imputers().get("EM")
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


plugin = EMPlugin
