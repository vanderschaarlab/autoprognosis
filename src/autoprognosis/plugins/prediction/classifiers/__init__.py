# stdlib
import glob
from os.path import basename, dirname, isfile, join

# autoprognosis absolute
from autoprognosis.plugins.core.base_plugin import PluginLoader
from autoprognosis.plugins.prediction.classifiers.base import (  # noqa: F401,E402
    ClassifierPlugin,
)

plugins = glob.glob(join(dirname(__file__), "plugin*.py"))


class Classifiers(PluginLoader):
    def __init__(self) -> None:
        super().__init__(plugins, ClassifierPlugin)


__all__ = [basename(f)[:-3] for f in plugins if isfile(f)] + [
    "Classifiers",
    "ClassifierPlugin",
]
