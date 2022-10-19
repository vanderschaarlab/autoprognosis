# stdlib
from typing import Any, List

# third party
import pytest

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
from autoprognosis.plugins.imputers import ImputerPlugin, Imputers
from autoprognosis.plugins.imputers.plugin_mean import plugin as mock_model


@pytest.fixture
def ctx() -> Imputers:
    return Imputers()


class Mock(ImputerPlugin):
    def __init__(self, **kwargs: Any) -> None:
        model = mock_model(**kwargs)

        super().__init__(model)

    @staticmethod
    def name() -> str:
        return "test"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return []


class Invalid:
    def __init__(self) -> None:
        pass


def test_load(ctx: Imputers) -> None:
    assert len(ctx._plugins) == 0
    ctx.get("mean")
    ctx.get("median")
    assert len(ctx._plugins) == 2
    assert len(ctx._plugins) == len(ctx)


def test_list(ctx: Imputers) -> None:
    ctx.get("mean")
    assert "mean" in ctx.list()


def test_add_get(ctx: Imputers) -> None:
    ctx.add("mock", Mock)

    assert "mock" in ctx.list()

    mock = ctx.get("mock")

    assert mock.name() == "test"

    ctx.reload()
    assert "mock" not in ctx.list()


def test_add_get_invalid(ctx: Imputers) -> None:
    with pytest.raises(ValueError):
        ctx.add("invalid", Invalid)

    assert "mock" not in ctx.list()

    with pytest.raises(ValueError):
        ctx.get("mock")


def test_iter(ctx: Imputers) -> None:
    for v in ctx:
        assert ctx[v].name() != ""
