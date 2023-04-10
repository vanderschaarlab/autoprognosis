# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any, List


class Hook(metaclass=ABCMeta):
    """AutoML hook interface.

    Methods:
        - cancel: True/False if to stop the current AutoML search.
        - heartbeat: Metrics/logs sink from the AutoML search
        - finish: executes after the AutoML search is finished.
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


class Hooks(list):
    """AutoML hooks container.

    Methods:
        - append: Add a hook to the container.
        - extend: Add multiple hooks to the container.
        - cancel: True/False if to stop the current AutoML search.
        - heartbeat: Metrics/logs sink from the AutoML search
        - finish: executes after the AutoML search is finished.
    """

    def append(self, hook: Hook) -> "Hooks":
        super().append(hook)
        return self

    def extend(self, hooks: List[Hook]) -> "Hooks":
        super().extend(hooks)
        return self

    def cancel(self) -> bool:
        return any(hook.cancel() for hook in self)

    def heartbeat(
        self, topic: str, subtopic: str, event_type: str, **kwargs: Any
    ) -> None:
        for hook in self:
            hook.heartbeat(topic, subtopic, event_type, **kwargs)

    def finish(self) -> None:
        for hook in self:
            hook.finish()

    def __repr__(self) -> str:
        return f"Hooks({super().__repr__()})"
