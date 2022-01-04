# stdlib
from typing import Any, List

# third party
from hyperimpute.plugins.imputers.plugin_hyperimpute import plugin as base_model

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.imputers.base as base


class HyperImputePlugin(base.ImputerPlugin):
    """HyperImpute strategy."""

    def __init__(self, **kwargs: Any) -> None:
        model = base_model(**kwargs)

        super().__init__(model)

    @staticmethod
    def name() -> str:
        return base_model.name()

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return base_model.hyperparameter_space()


plugin = HyperImputePlugin
