# stdlib
from typing import Any, List

# third party
from hyperimpute.plugins.imputers.plugin_gain import plugin as base_model

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.imputers.base as base


class GainPlugin(base.ImputerPlugin):
    """GAIN Imputation for static data using Generative Adversarial Nets.
     The training steps are:
      - The generato imputes the missing components conditioned on what is actually observed, and outputs a           completed vector.
      - The discriminator takes a completed vector and attempts to determine which components were actually observed  and which were imputed.

     Args:

         batch_size: int
             The batch size for the training steps.
         n_epochs: int
             Number of epochs for training.
         hint_rate: float
             Percentage of additional information for the discriminator.
         loss_alpha: int
             Hyperparameter for the generator loss.

     Paper: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets,  " ICML, 2018.
     Original code: https://github.com/jsyoon0823/GAIN


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
