# stdlib
from typing import Any

# first party
import wandb

# autoprognosis relative
from .base import Hooks


class WandbHooks(Hooks):
    """AutoML hooks interface.

    Methods:
        - cancel: True/False if to stop the current AutoML search.
        - heartbeat: Metrics/logs sink from the AutoML search

    """

    def __init__(self, project: str) -> None:
        print("Create project")
        self.project = project
        wandb.login()
        self.cbk = wandb.init(
            project=project,
            name=f"autoprognosis_study_{project}",
            resume=True,
            reinit=True,
            group="DDP",
        )

    def cancel(self) -> bool:
        return False

    def heartbeat(
        self, topic: str, subtopic: str, event_type: str, **kwargs: Any
    ) -> None:
        print(topic, subtopic, event_type, kwargs)
        # self.cbk.log(kwargs)

    def finish(self) -> None:
        self.cbk.finish()
