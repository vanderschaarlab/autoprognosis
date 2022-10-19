# stdlib
from typing import Any

# autoprognosis absolute
from autoprognosis.hooks import Hooks
import autoprognosis.logger as log


class DefaultHooks(Hooks):
    """Default hook used by the explorers"""

    def cancel(self) -> bool:
        return False

    def heartbeat(
        self, topic: str, subtopic: str, event_type: str, **kwargs: Any
    ) -> None:
        log.debug(f"[{topic}][{subtopic}] {event_type}")
