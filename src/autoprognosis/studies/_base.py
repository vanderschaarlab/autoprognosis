# stdlib
from abc import ABCMeta, abstractmethod
from typing import Any


class Study(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self) -> Any:
        ...
