# stdlib
from typing import Any, List

# third party
import pandas as pd
import pytest

# adjutorium absolute
from adjutorium.plugins.preprocessors import PreprocessorPlugin, Preprocessors


@pytest.fixture
def ctx() -> Preprocessors:
    return Preprocessors()


class Mock(PreprocessorPlugin):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def name() -> str:
        return "test"

    @staticmethod
    def subtype() -> str:
        return "feature_scaling"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Any]:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "Mock":
        return self

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return {}

    def save(self) -> bytes:
        return b""

    @classmethod
    def load(cls, buff: bytes) -> "Mock":
        return cls()


class Invalid:
    def __init__(self) -> None:
        pass


def test_load(ctx: Preprocessors) -> None:
    assert len(ctx._plugins) > 0
    assert len(ctx._plugins) == len(ctx)


def test_list(ctx: Preprocessors) -> None:
    assert "nop" in ctx.list()


def test_add_get(ctx: Preprocessors) -> None:
    ctx.add("mock", Mock)

    assert "mock" in ctx.list()

    mock = ctx.get("mock")

    assert mock.name() == "test"


def test_add_get_invalid(ctx: Preprocessors) -> None:
    with pytest.raises(ValueError):
        ctx.add("invalid", Invalid)

    assert "mock" not in ctx.list()

    with pytest.raises(ValueError):
        ctx.get("mock")


def test_iter(ctx: Preprocessors) -> None:
    for v in ctx:
        assert ctx[v].name() != ""
