# stdlib
from importlib.abc import Loader
import glob
import importlib.util
from os.path import basename, dirname, isfile, join
from typing import Any, Dict, Generator, List, Type

# adjutorium absolute
import adjutorium.logger as log

# adjutorium relative
from .base import PreprocessorPlugin  # noqa: F401,E402

feature_scaling_plugins = glob.glob(
    join(dirname(__file__), "feature_scaling/plugin*.py")
)
dim_reduction_plugins = glob.glob(
    join(dirname(__file__), "dimensionality_reduction/plugin*.py")
)


class Preprocessors:
    def __init__(self, category: str = "feature_scaling") -> None:
        assert category in ["feature_scaling", "dimensionality_reduction"]

        self.category = category
        self._plugins: Dict[str, Type] = {}

        self._load_default_plugins(category)

    def _load_default_plugins(self, category: str) -> None:
        if category == "feature_scaling":
            plugins = feature_scaling_plugins
        elif category == "dimensionality_reduction":
            plugins = dim_reduction_plugins
        else:
            raise ValueError(f"invalid preprocessing category {category}")

        for plugin in plugins:
            name = basename(plugin)
            spec = importlib.util.spec_from_file_location(name, plugin)
            assert isinstance(spec.loader, Loader)

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            try:
                cls = mod.plugin  # type: ignore
            except BaseException as e:
                log.critical(f"module {name} load failed {e}")
                continue

            log.debug(f"Loaded plugin {cls.type()} - {cls.name()}")
            self.add(cls.name(), cls)

    def list(self) -> List[str]:
        return list(self._plugins.keys())

    def types(self) -> List[Type]:
        return list(self._plugins.values())

    def add(self, name: str, cls: Type) -> "Preprocessors":
        if name in self._plugins:
            raise ValueError(f"Plugin {name} already exists.")

        if not issubclass(cls, PreprocessorPlugin):
            raise ValueError(
                f"Plugin {name} must derive the PreprocessorPlugin interface."
            )

        self._plugins[name] = cls

        return self

    def get(self, name: str, *args: Any, **kwargs: Any) -> PreprocessorPlugin:
        if name not in self._plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        return self._plugins[name](*args, **kwargs)

    def get_type(self, name: str) -> Type:
        if name not in self._plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        return self._plugins[name]

    def __iter__(self) -> Generator:
        for x in self._plugins:
            yield x

    def __len__(self) -> int:
        return len(self.list())

    def __getitem__(self, key: str) -> PreprocessorPlugin:
        return self.get(key)


__all__ = (
    [basename(f)[:-3] for f in feature_scaling_plugins if isfile(f)]
    + [basename(f)[:-3] for f in dim_reduction_plugins if isfile(f)]
    + [
        "Preprocessors",
        "PreprocessorPlugin",
    ]
)
