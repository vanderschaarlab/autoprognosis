# stdlib
from typing import Any, List

# third party
from hyperimpute.plugins.imputers.plugin_mice import plugin as base_model

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.imputers.base as base


class MicePlugin(base.ImputerPlugin):
    """Imputation plugin for completing missing values using the Multivariate Iterative chained equations and multiple imputations.

    Method:
        Multivariate Iterative chained equations(MICE) methods model each feature with missing values as a function of other features in a round-robin fashion. For each step of the round-robin imputation, we use a BayesianRidge estimator, which does a regularized linear regression.
        The class `sklearn.impute.IterativeImputer` is able to generate multiple imputations of the same incomplete dataset. We can then learn a regression or classification model on different imputations of the same dataset.
        Setting `sample_posterior=True` for the IterativeImputer will randomly draw values to fill each missing value from the Gaussian posterior of the predictions. If each `IterativeImputer` uses a different `random_state`, this results in multiple imputations, each of which can be used to train a predictive model.
        The final result is the average of all the `n_imputation` estimates.

    Args:
        n_imputations: int, default=5i
            number of multiple imputations to perform.
        max_iter: int, default=500
            maximum number of imputation rounds to perform.
        random_state: int, default set to the current time.
            seed of the pseudo random number generator to use.

    Example:
        >>> import numpy as np
        >>> from autoprognosis.plugins.imputers import Imputers
        >>> plugin = Imputers().get("mice")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
                  0        1         2         3
        0  1.000000  1.00000  1.000000  1.000000
        1  1.222412  1.68686  1.687483  1.221473
        2  1.000000  2.00000  2.000000  1.000000
        3  2.000000  2.00000  2.000000  2.000000
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


plugin = MicePlugin
