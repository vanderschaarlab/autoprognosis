# stdlib
from typing import Any, Dict

import wandb

# autoprognosis absolute
import autoprognosis.logger as log

# autoprognosis relative
from .base import Hook


class WandbHook(Hook):
    def __init__(self, **config: Any):
        super().__init__()
        self.wandb_config = config

    def cancel(self) -> bool:
        return False

    def heartbeat(
        self, topic: str, subtopic: str, event_type: str, **kwargs: Any
    ) -> None:
        if wandb.run is None:
            wandb.init(
                project='autoprognosis',
                group=topic,
                name=f'{topic}-{subtopic}',
                job_type=event_type,
                **self.wandb_config
            )
        table = {k: kwargs.pop(k) for k in list(kwargs)
                 if isinstance(kwargs[k], str)}
        table = wandb.Table(columns=list(table), data=[list(table.values())])
        kwargs['text_table'] = table
        wandb.log(kwargs)

    def finish(self) -> None:
        pass
