# stdlib
from typing import Any, Callable, Dict, Tuple, Type

# third party
import numpy as np
from optuna.trial import Trial
import pandas as pd

# autoprognosis absolute
import autoprognosis.plugins.utils.decorators as decorators
import autoprognosis.utils.serialization as serialization


def _generate_name_impl(plugins: Tuple[Type, ...]) -> Callable:
    def name_impl(*args: Any) -> str:
        return "->".join(p.name() for p in plugins)

    return name_impl


def _generate_type_impl(plugins: Tuple[Type, ...]) -> Callable:
    def type_impl(*args: Any) -> str:
        return "->".join(p.type() for p in plugins)

    return type_impl


def _generate_hyperparameter_space_impl(plugins: Tuple[Type, ...]) -> Callable:
    def hyperparameter_space_impl(*args: Any, **kwargs: Any) -> Dict:
        out = {}
        for p in plugins:
            out[p.name()] = p.hyperparameter_space(*args, **kwargs)
        return out

    return hyperparameter_space_impl


def _generate_hyperparameter_space_for_layer_impl(
    plugins: Tuple[Type, ...]
) -> Callable:
    def hyperparameter_space_for_layer_impl(
        layer: str, *args: Any, **kwargs: Any
    ) -> Dict:
        for p in plugins:
            if p.name() == layer:
                return p.hyperparameter_space(*args, **kwargs)
        raise ValueError(f"invalid layer {layer}")

    return hyperparameter_space_for_layer_impl


def _generate_sample_param_impl(plugins: Tuple[Type, ...]) -> Callable:
    def sample_param_impl(trial: Trial, *args: Any, **kwargs: Any) -> Dict:
        sample: dict = {}
        for p in plugins:
            sample[p.name()] = p.sample_hyperparameters(trial)

        return sample

    return sample_param_impl


def _generate_constructor() -> Callable:
    def _sanity_checks(plugins: Tuple[Type, ...]) -> None:
        if len(plugins) == 0:
            raise RuntimeError("invalid empty pipeline.")

        predictors = 0
        for pl in plugins:
            if pl.type() == "prediction":
                predictors += 1

        if predictors != 1 or plugins[-1].type() != "prediction":
            raise RuntimeError("The last plugin of a pipeling must be a predictor")

    def init_impl(self: Any, args: dict = {}, output: str = "pandas") -> None:
        _sanity_checks(self.plugin_types)

        self.stages = []
        self.args = args

        if output not in ["pandas", "numpy"]:
            raise RuntimeError("Invalid output type")
        if output == "pandas":
            self.output = pd.DataFrame
        elif output == "numpy":
            self.output = np.asarray

        for plugin_type in self.plugin_types:
            plugin_args = {}
            if plugin_type.name() in args:
                plugin_args = args[plugin_type.name()]
            self.stages.append(plugin_type(**plugin_args))

    return init_impl


def _generate_change_output() -> Callable:
    def change_output_impl(self: Any, output: str) -> None:
        if output not in ["pandas", "numpy"]:
            raise RuntimeError("Invalid output type")
        if output == "pandas":
            self.output = pd.DataFrame
        elif output == "numpy":
            self.output = np.asarray

    return change_output_impl


def _generate_get_args() -> Callable:
    def get_args_impl(self: Any) -> Dict:
        return self.args

    return get_args_impl


def _generate_fit() -> Callable:
    def fit_impl(self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        local_X = X.copy()
        for stage in self.stages[:-1]:
            local_X = pd.DataFrame(local_X)
            local_X = stage.fit_transform(local_X)

        self.stages[-1].fit(local_X, *args, **kwargs)

        return self

    return fit_impl


def _generate_is_fitted() -> Callable:
    def fit_impl(self: Any) -> Any:
        return self.stages[-1].is_fitted()

    return fit_impl


def _generate_predict() -> Callable:
    @decorators.benchmark
    def predict_impl(
        self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        local_X = X.copy()
        for stage in self.stages[:-1]:
            local_X = stage.transform(local_X)

        result = self.stages[-1].predict(local_X, *args, **kwargs)

        return self.output(result)

    return predict_impl


def _generate_predict_proba() -> Callable:
    @decorators.benchmark
    def predict_proba_impl(
        self: Any, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        local_X = X.copy()
        for stage in self.stages[:-1]:
            local_X = stage.transform(local_X, *args, **kwargs)

        result = self.stages[-1].predict_proba(local_X)

        if result.isnull().values.any():
            raise ValueError(
                "pipeline ({})({}) failed: nan in predict_proba output".format(
                    self.name(), self.get_args()
                )
            )
        return self.output(result)

    return predict_proba_impl


def _generate_score() -> Callable:
    @decorators.benchmark
    def predict_score(self: Any, X: pd.DataFrame, y: pd.DataFrame) -> float:
        local_X = X.copy()
        for stage in self.stages[:-1]:
            local_X = stage.transform(local_X)

        return self.stages[-1].score(local_X, y)

    return predict_score


def _generate_save_template() -> Callable:
    def save_template_impl(self: Any) -> bytes:
        plugins = []
        for plugin in self.plugin_types:
            plugins.append(plugin.fqdn())

        buff = {
            "args": self.args,
            "plugins": plugins,
            "name": self.name(),
        }
        return serialization.save(buff)

    return save_template_impl


def _generate_save() -> Callable:
    def save_impl(self: Any) -> bytes:
        plugins = []
        for stage in self.stages:
            plugins.append(stage.fqdn())

        buff = self.__dict__.copy()

        buff["plugins"] = plugins

        return serialization.save(buff)

    return save_impl


def _generate_load_template() -> Callable:
    def load_template_impl(buff: bytes) -> dict:
        return serialization.load(buff)

    return load_template_impl


def _generate_load() -> Callable:
    def load_impl(buff: bytes) -> dict:
        return serialization.load(buff)

    return load_impl


def _generate_setstate() -> Callable:
    def setstate_impl(self: Any, state: dict) -> Any:
        repr_ = serialization.load(state["object"])

        self.__dict__ = repr_

        return self

    return setstate_impl


def _generate_getstate() -> Callable:
    def getstate_impl(self: Any) -> dict:
        return {"object": self.save()}

    return getstate_impl


__all__ = [
    "_generate_name_impl",
    "_generate_type_impl",
    "_generate_hyperparameter_space_impl",
    "_generate_hyperparameter_space_for_layer_impl",
    "_generate_sample_param_impl",
    "_generate_constructor",
    "_generate_fit",
    "_generate_is_fitted",
    "_generate_predict",
    "_generate_predict_proba",
    "_generate_score",
    "_generate_get_args",
    "_generate_load_template",
    "_generate_load",
    "_generate_save_template",
    "_generate_save",
    "_generate_setstate",
    "_generate_getstate",
    "_generate_change_output",
]
