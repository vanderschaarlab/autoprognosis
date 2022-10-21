# stdlib
import glob
from os.path import basename, dirname, isfile, join

# autoprognosis absolute
from autoprognosis.plugins.core.base_plugin import PluginLoader

# autoprognosis relative
from .base import UncertaintyPlugin  # noqa: F401,E402

plugins = glob.glob(join(dirname(__file__), "plugin*.py"))


class UncertaintyQuantification(PluginLoader):
    def __init__(self) -> None:
        super().__init__(plugins, UncertaintyPlugin)


__all__ = [basename(f)[:-3] for f in plugins if isfile(f)] + [
    "UncertaintyPlugin",
    "UncertaintyQuantification",
]
