# stdlib
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# autoprognosis absolute
from autoprognosis.plugins.explainers import Explainers  # noqa: F401,E402
from autoprognosis.plugins.imputers import Imputers
from autoprognosis.plugins.prediction import Predictions
from autoprognosis.plugins.preprocessors import Preprocessors
import autoprognosis.plugins.utils  # noqa: F401,E402

# autoprognosis relative
from .core import base_plugin  # noqa: F401,E402


class Plugins:
    def __init__(self) -> None:
        self._plugins: Dict[
            str, Dict[str, Union[Imputers, Predictions, Preprocessors, Explainers]]
        ] = {
            "imputer": {
                "default": Imputers(),
            },
            "prediction": {
                "classifier": Predictions(category="classifier"),
                "regression": Predictions(category="regression"),
                "risk_estimation": Predictions(category="risk_estimation"),
            },
            "explainer": {
                "default": Explainers(),
            },
            "preprocessor": {
                "feature_scaling": Preprocessors(category="feature_scaling"),
                "dimensionality_reduction": Preprocessors(
                    category="dimensionality_reduction"
                ),
            },
        }

    def list(self) -> dict:
        res: Dict[str, Dict[str, List[str]]] = {}
        for src in self._plugins:
            res[src] = {}
            for subtype in self._plugins[src]:
                res[src][subtype] = self._plugins[src][subtype].list()
        return res

    def list_available(self) -> dict:
        res: Dict[str, Dict[str, List[str]]] = {}
        for src in self._plugins:
            res[src] = {}
            for subtype in self._plugins[src]:
                res[src][subtype] = self._plugins[src][subtype].list_available()
        return res

    def add(self, cat: str, subtype: str, name: str, cls: Type) -> "Plugins":
        self._plugins[cat][subtype].add(name, cls)

        return self

    def get(self, cat: str, subtype: str, name: str, *args: Any, **kwargs: Any) -> Any:
        return self._plugins[cat][subtype].get(name, *args, **kwargs)

    def get_type(self, cat: str, subtype: str, name: str) -> Type:
        return self._plugins[cat][subtype].get_type(name)

    def get_any_type(self, name: str) -> Optional[Type]:
        for cat in self._plugins:
            for subcat in self._plugins[cat]:
                try:
                    loaded = self._plugins[cat][subcat].get_type(name)
                    return loaded
                except BaseException:
                    continue
        return None


def group(names: List[str]) -> Tuple[Type, ...]:
    res = []

    plugins = Plugins()
    for fqdn in names:
        if "." not in fqdn:
            raise RuntimeError("invalid fqdn")

        cat, subtype, name = fqdn.split(".")

        res.append(plugins.get_type(cat, subtype, name))

    return tuple(res)
