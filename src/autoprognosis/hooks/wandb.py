# stdlib
from typing import Any

# third party
import wandb

# autoprognosis relative
from .base import Hook


class WandbHook(Hook):
    def __init__(self, **config: Any) -> None:
        super().__init__()
        self.wandb_config = config

    def cancel(self) -> bool:
        return False

    def heartbeat(
        self, topic: str, subtopic: str, event_type: str, **kwargs: Any
    ) -> None:
        if wandb.run is None:
            wandb.init(
                project="autoprognosis",
                group=topic,
                name=f"{topic}-{subtopic}",
                job_type=event_type,
                **self.wandb_config,
            )
        table = {k: kwargs.pop(k) for k in list(kwargs) if isinstance(kwargs[k], str)}
        table = wandb.Table(columns=list(table), data=[list(table.values())])
        kwargs["text_table"] = table
        wandb.log(kwargs)

    def finish(self) -> None:
        pass
