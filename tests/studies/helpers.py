# stdlib
import datetime
from typing import Any

# autoprognosis absolute
from autoprognosis.hooks import Hooks


class MockHook(Hooks):
    def __init__(self) -> None:
        self._started_at = datetime.datetime.utcnow()

    def cancel(self) -> bool:
        # cancel after 10 seconds
        time_passed = datetime.datetime.utcnow() - self._started_at

        return time_passed.total_seconds() > 10

    def heartbeat(
        self, topic: str, subtopic: str, event_type: str, **kwargs: Any
    ) -> None:
        pass

    def finish(self) -> None:
        pass
