# stdlib
from typing import Any

# autoprognosis absolute
import autoprognosis.logger as log

# autoprognosis relative
from .base import Hook, Hooks


class DefaultHook(Hook):
    def cancel(self) -> bool:
        return False

    def heartbeat(
        self, topic: str, subtopic: str, event_type: str, **kwargs: Any
    ) -> None:
        log.debug(f"[{topic}][{subtopic}] {event_type}")

    def finish(self) -> None:
        pass


class DefaultHooks(Hooks):
    def __init__(self) -> None:
        super().__init__([DefaultHook()])
