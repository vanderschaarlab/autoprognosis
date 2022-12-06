# stdlib
from typing import Any, List

# third party
from hyperimpute.plugins.imputers.plugin_nop import plugin as base_model

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.imputers.base as base


class NopPlugin(base.ImputerPlugin):
    """Imputer plugin that doesn't alter the dataset."""

    def __init__(self, random_state: int = 0, **kwargs: Any) -> None:
        model = base_model(random_state=random_state, **kwargs)

        super().__init__(model)

    @staticmethod
    def name() -> str:
        return base_model.name()

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return base_model.hyperparameter_space()


plugin = NopPlugin
