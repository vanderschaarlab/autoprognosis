# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any


class Hooks(metaclass=ABCMeta):
    """AutoML hooks interface.

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
