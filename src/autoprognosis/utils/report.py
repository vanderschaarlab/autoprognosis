# stdlib
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Union

# third party
from torch.utils.tensorboard import SummaryWriter

# autoprognosis absolute
import autoprognosis.logger as log


class ReportStub(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def add_scalar(self, name: str, val: Union[str, int, float], step: int) -> None:
        ...

    @abstractmethod
    def add_scalars(self, name: str, val: Dict, step: int) -> None:
        ...

    @abstractmethod
    def add_text(self, name: str, val: str, step: int) -> None:
        ...

    @abstractmethod
    def add_pr_curve(self, *args: Any, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def add_custom_scalars(self, *args: Any, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def add_hparams(self, *args: Any, **kwargs: Any) -> None:
        ...


class TensorboardLogger(ReportStub):
    def __init__(self, name: str) -> None:
        self.writer = SummaryWriter(
            f"runs/experiments_{name}_"
            + datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
        )

    def add_scalar(self, name: str, val: Union[str, int, float], step: int) -> None:
        self.writer.add_scalar(name, val, step)

    def add_scalars(self, name: str, val: Dict, step: int) -> None:
        self.writer.add_scalars(name, val, step)

    def add_text(self, name: str, val: str, step: int) -> None:
        self.writer.add_text(name, val, step)

    def add_pr_curve(self, *args: Any, **kwargs: Any) -> None:
        self.writer.add_pr_curve(*args, **kwargs)

    def add_custom_scalars(self, *args: Any, **kwargs: Any) -> None:
        self.writer.add_custom_scalars(*args, **kwargs)

    def add_hparams(self, *args: Any, **kwargs: Any) -> None:
        self.writer.add_hparams(*args, **kwargs)


class SimpleLogger(ReportStub):
    def __init__(self, name: str) -> None:
        self.name = name

    def add_scalar(self, name: str, val: Union[str, int, float], step: int) -> None:
        log.debug(f"[self.name] Iter {step}: {name} = {val}")

    def add_scalars(self, name: str, val: Dict, step: int) -> None:
        for k in val:
            log.debug(f"[self.name] Iter {step}: {k} = {val[k]}")

    def add_text(self, name: str, val: str, step: int) -> None:
        log.debug(f"[self.name] Iter {step}: {name} = {val}")

    def add_pr_curve(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_custom_scalars(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_hparams(self, *args: Any, **kwargs: Any) -> None:
        pass


class Report(ReportStub):
    """Helper class for reporting AutoPrognosis metrics to different backends.
    Supported outputs:
        - standard logging
        - Tensorboard
    """

    def __init__(
        self,
        name: str,
        logger: bool = True,
        tensorboard: bool = True,
        log_hyperparams: bool = False,
    ):

        self.log_hyperparams = log_hyperparams
        self.sinks: List[ReportStub] = []
        if logger:
            self.sinks.append(SimpleLogger(name))

        if tensorboard:
            self.sinks.append(TensorboardLogger(name))

    def add_scalar(self, name: str, val: Union[str, int, float], step: int) -> None:
        for sink in self.sinks:
            sink.add_scalar(name, val, step)

    def add_scalars(self, name: str, val: Dict, step: int) -> None:
        for sink in self.sinks:
            sink.add_scalars(name, val, step)

    def add_text(self, name: str, val: str, step: int) -> None:
        for sink in self.sinks:
            sink.add_text(name, val, step)

    def add_pr_curve(self, *args: Any, **kwargs: Any) -> None:
        for sink in self.sinks:
            sink.add_pr_curve(*args, **kwargs)

    def add_custom_scalars(self, *args: Any, **kwargs: Any) -> None:
        for sink in self.sinks:
            sink.add_custom_scalars(*args, **kwargs)

    def add_hparams(self, *args: Any, **kwargs: Any) -> None:
        if not self.log_hyperparams:
            return

        for sink in self.sinks:
            sink.add_hparams(*args, **kwargs)
