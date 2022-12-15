# stdlib
from typing import Any, List

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.imputers.base as base
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        from hyperimpute.plugins.imputers.plugin_softimpute import plugin as base_model

        break
    except ImportError:
        depends = ["hyperimpute"]
        install(depends)


class SoftImputePlugin(base.ImputerPlugin):
    """The SoftImpute algorithm fits a low-rank matrix approximation to a matrix with missing values via nuclear-    norm regularization. The algorithm can be used to impute quantitative data.
     To calibrate the the nuclear-norm regularization parameter(shrink_lambda), we perform cross-                     validation(_cv_softimpute)

     Args:
         maxit: int, default=500
             maximum number of imputation rounds to perform.
         convergence_threshold : float, default=1e-5
             Minimum ration difference between iterations before stopping.
         max_rank : int, default=2
             Perform a truncated SVD on each iteration with this value as its rank.
         shrink_lambda: float, default=0
             Value by which we shrink singular values on each iteration. If it's missing, it is calibrated using      cross validation.
         cv_len: int, default=15
             the length of the grid on which the cross-validation is performed.

    Example:
        >>> import numpy as np
        >>> from autoprognosis.plugins.imputers import Imputers
        >>> plugin = Imputers().get("softimpute")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])
                      0             1             2             3
        0  1.000000e+00  1.000000e+00  1.000000e+00  1.000000e+00
        1  3.820605e-16  1.708249e-16  1.708249e-16  3.820605e-16
        2  1.000000e+00  2.000000e+00  2.000000e+00  1.000000e+00
        3  2.000000e+00  2.000000e+00  2.000000e+00  2.000000e+00

     Reference: "Spectral Regularization Algorithms for Learning Large Incomplete Matrices", by Mazumder, Hastie,    and Tibshirani.
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


plugin = SoftImputePlugin
