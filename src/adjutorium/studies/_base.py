# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any

# adjutorium absolute
from adjutorium.hooks import Hooks
import adjutorium.logger as log


class DefaultHooks(Hooks):
    def cancel(self) -> bool:
        return False

    def heartbeat(
        self, topic: str, subtopic: str, event_type: str, **kwargs: Any
    ) -> None:
        log.debug(f"[{topic}][{subtopic}] {event_type}")


class Study(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self) -> Any:
        ...
