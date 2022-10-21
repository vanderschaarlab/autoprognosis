# stdlib
import glob
from os.path import basename, dirname, isfile, join

# autoprognosis absolute
from autoprognosis.plugins.core.base_plugin import PluginLoader

# autoprognosis relative
from .base import ExplainerPlugin  # noqa: F401,E402

plugins = glob.glob(join(dirname(__file__), "plugin*.py"))


class Explainers(PluginLoader):
    def __init__(self) -> None:
        super().__init__(plugins, ExplainerPlugin)


__all__ = [basename(f)[:-3] for f in plugins if isfile(f)] + [
    "Explainers",
    "ExplainerPlugin",
]
