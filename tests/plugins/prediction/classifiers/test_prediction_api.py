# stdlib
from typing import Any, List

# third party
import pandas as pd
import pytest

# autoprognosis absolute
from autoprognosis.plugins.prediction import Predictions
from autoprognosis.plugins.prediction.classifiers import ClassifierPlugin


@pytest.fixture
def ctx() -> Predictions:
    return Predictions()


class Mock(ClassifierPlugin):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def name() -> str:
        return "test"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List:
        return []

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "Mock":
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return {}

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return {}

    def save(self) -> bytes:
        return b""

    @classmethod
    def load(cls, buff: bytes) -> "Mock":
        return cls()


class Invalid:
    def __init__(self) -> None:
        pass


def test_load(ctx: Predictions) -> None:
    assert len(ctx._plugins) == 0
    ctx.get("xgboost")
    assert len(ctx._plugins) == 1


def test_list(ctx: Predictions) -> None:
    ctx.get("bagging")
    assert "bagging" in ctx.list()
    assert "catboost" not in ctx.list()


def test_add_get(ctx: Predictions) -> None:
    ctx.add("mock", Mock)

    assert "mock" in ctx.list()

    mock = ctx.get("mock")

    assert mock.name() == "test"

    ctx.reload()
    assert "mock" not in ctx.list()


def test_add_get_invalid(ctx: Predictions) -> None:
    with pytest.raises(ValueError):
        ctx.add("invalid", Invalid)

    assert "mock" not in ctx.list()

    with pytest.raises(ValueError):
        ctx.get("mock")


def test_iter(ctx: Predictions) -> None:
    for v in ctx:
        assert ctx[v].name() != ""
