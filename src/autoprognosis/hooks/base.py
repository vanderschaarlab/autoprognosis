# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, List


class Hook(metaclass=ABCMeta):
    """AutoML hook interface.

    Methods:
        - cancel: True/False if to stop the current AutoML search.
        - heartbeat: Metrics/logs sink from the AutoML search

    """

    @abstractmethod
    def cancel(self) -> bool:
        ...

    @abstractmethod
    def heartbeat(
        self, topic: str, subtopic: str, event_type: str, **kwargs: Any
    ) -> None:
        ...

    @abstractmethod
    def finish(self) -> None:
        ...


class Hooks:
    """AutoML hooks container.

    Methods:
        - cancel: True/False if to stop the current AutoML search.
        - heartbeat: Metrics/logs sink from the AutoML search

    """
    def __init__(self, hooks: List[Hook]) -> None:
        self.hooks = hooks
        
    def append(self, hook: Hook) -> "Hooks":
        self.hooks.append(hook)
        return self

    def cancel(self) -> bool:
        return any(hook.cancel() for hook in self.hooks)

    def heartbeat(
        self, topic: str, subtopic: str, event_type: str, **kwargs: Any
    ) -> None:
        for hook in self.hooks:
            hook.heartbeat(topic, subtopic, event_type, **kwargs)

    def finish(self) -> None:
        for hook in self.hooks:
            hook.finish()
