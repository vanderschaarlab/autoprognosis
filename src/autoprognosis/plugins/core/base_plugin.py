# stdlib
from abc import ABCMeta, abstractmethod
from importlib.abc import Loader
import importlib.util
from os.path import basename
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Type

# third party
import numpy as np
from optuna.trial import Trial
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# autoprognosis absolute
import autoprognosis.logger as log
import autoprognosis.plugins.utils.cast as cast

# autoprognosis relative
from .params import Params


class Plugin(metaclass=ABCMeta):
    """Base class for all plugins.
    Each derived class must implement the following methods:

        - type() - a static method that returns the type of the plugin. e.g., imputation, preprocessing, prediction, etc.
        - subtype() - optional method that returns the subtype of the plugin. e.g. Potential subtypes:

            - preprocessing: feature_scaling, dimensionality reduction
            - prediction: classifiers, prediction, survival analysis
        - name() - a static method that returns the name of the plugin. e.g., EM, mice, etc.
        - hyperparameter_space() - a static method that returns the hyperparameters that can be tuned during the optimization. The method will return a list of `Params` derived objects.
        - _fit() - internal method, called by `fit` on each training set.
        - _transform() - internal method, called by `transform`. Used by imputation or preprocessing plugins.
        - _predict() - internal method, called by `predict`. Used by classification/prediction plugins.
        - load/save - serialization methods

    If any method implementation is missing, the class constructor will fail.
    """

    def __init__(self) -> None:
        self.output = pd.DataFrame
        self._backup_encoders: Optional[Dict[str, LabelEncoder]] = {}
        self._drop_features: Optional[List[str]] = []
        self._fitted = False

    def change_output(self, output: str) -> None:
        if output not in ["pandas", "numpy"]:
            raise RuntimeError("Invalid output type")
        if output == "pandas":
            self.output = pd.DataFrame
        elif output == "numpy":
            self.output = np.asarray

    @staticmethod
    @abstractmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[Params]:
        """The hyperparameter search domain, used for tuning."""
        ...

    @classmethod
    def sample_hyperparameters(
        cls, trial: Trial, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Sample hyperparameters for Optuna."""
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample(trial)

        return results

    @classmethod
    def sample_hyperparameters_np(
        cls, random_state: int = 0, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Sample hyperparameters as a dict."""
        param_space = cls.hyperparameter_space(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample_np()

        return results

    @classmethod
    def hyperparameter_space_fqdn(cls, *args: Any, **kwargs: Any) -> List[Params]:
        """The hyperparameter domain using they fully-qualified name."""
        res = []
        for param in cls.hyperparameter_space(*args, **kwargs):
            fqdn_param = param
            fqdn_param.name = (
                cls.type() + "." + cls.subtype() + "." + cls.name() + "." + param.name
            )
            res.append(fqdn_param)

        return res

    @classmethod
    def sample_hyperparameters_fqdn(
        cls, trial: Trial, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Sample hyperparameters using they fully-qualified name."""
        param_space = cls.hyperparameter_space_fqdn(*args, **kwargs)

        results = {}

        for hp in param_space:
            results[hp.name] = hp.sample(trial)

        return results

    @staticmethod
    @abstractmethod
    def name() -> str:
        """The name of the plugin, e.g.: xgboost"""
        ...

    @staticmethod
    @abstractmethod
    def type() -> str:
        """The type of the plugin, e.g.: prediction"""
        ...

    @staticmethod
    @abstractmethod
    def subtype() -> str:
        """The type of the plugin, e.g.: classifier"""
        ...

    @classmethod
    def fqdn(cls) -> str:
        """The fully-qualified name of the plugin: type->subtype->name"""
        return cls.type() + "." + cls.subtype() + "." + cls.name()

    def is_fitted(self) -> bool:
        """Check if the model was trained"""
        try:
            return self._fitted
        except BaseException:
            return True  # backwards compatible

    def fit_transform(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Fit the model and transform the training data. Used by imputers and preprocessors."""
        return pd.DataFrame(self.fit(X, *args, *kwargs).transform(X))

    def fit_predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Fit the model and predict the training data. Used by predictors."""
        return pd.DataFrame(self.fit(X, *args, *kwargs).predict(X))

    def _preprocess_training_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode the input"""
        X = cast.to_dataframe(X).copy()
        self._backup_encoders = {}

        for col in X.columns:
            if X[col].dtype.name not in ["object", "category"]:
                continue

            values = list(X[col].unique())
            values.append("unknown")
            encoder = LabelEncoder().fit(values)
            X.loc[X[col].notna(), col] = encoder.transform(X[col][X[col].notna()])

            self._backup_encoders[col] = encoder
        return X

    def _preprocess_inference_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode the inference input"""
        X = cast.to_dataframe(X).copy()

        if self._backup_encoders is None:
            self._backup_encoders = {}

        for col in self._backup_encoders:
            eval_data = X[col][X[col].notna()]
            inf_values = [
                x if x in self._backup_encoders[col].classes_ else "unknown"
                for x in eval_data
            ]

            X.loc[X[col].notna(), col] = self._backup_encoders[col].transform(
                inf_values
            )

        return X

    def fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "Plugin":
        """Train the plugin

        Args:
            X: pd.DataFrame
        """
        X = self._preprocess_training_data(X)

        log.debug(f"Training {self.fqdn()}, input shape = {X.shape}")
        self._fit(X, *args, **kwargs)
        log.debug(f"Done Training {self.fqdn()}, input shape = {X.shape}")

        self._fitted = True

        return self

    @abstractmethod
    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "Plugin":
        ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input. Used by imputers and preprocessors.

        Args:
            X: pd.DataFrame

        """
        if not self.is_fitted():
            raise RuntimeError("Fit the model first")
        X = self._preprocess_inference_data(X)
        log.debug(f"Transforming using {self.fqdn()}, input shape = {X.shape}")
        return self.output(self._transform(X))

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Run predictions for the input. Used by predictors.

        Args:
            X: pd.DataFrame

        """
        if not self.is_fitted():
            raise RuntimeError("Fit the model first")
        X = self._preprocess_inference_data(X)
        log.debug(f"Predicting using {self.fqdn()}, input shape = {X.shape}")
        return self.output(self._predict(X, *args, *kwargs))

    @abstractmethod
    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        ...

    @abstractmethod
    def save(self) -> bytes:
        """Save the plugin to bytes"""
        ...

    @classmethod
    @abstractmethod
    def load(cls, buff: bytes) -> "Plugin":
        """Load the plugin from bytes"""
        ...


class PluginLoader:
    def __init__(self, plugins: list, expected_type: Type) -> None:
        self._plugins: Dict[str, Type] = {}
        self._available_plugins = {}
        for plugin in plugins:
            stem = Path(plugin).stem.split("plugin_")[-1]
            self._available_plugins[stem] = plugin

        self._expected_type = expected_type

    def _load_single_plugin(self, plugin: str) -> None:
        name = basename(plugin)
        failed = False
        for retry in range(2):
            try:
                spec = importlib.util.spec_from_file_location(name, plugin)
                if not isinstance(spec.loader, Loader):
                    raise RuntimeError("invalid plugin type")

                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                cls = mod.plugin  # type: ignore
                failed = False
                break
            except BaseException as e:
                log.critical(f"load failed: {e}")
                failed = True
                break

        if failed:
            log.critical(f"module {name} load failed")
            return

        log.debug(f"Loaded plugin {cls.type()} - {cls.name()}")
        self.add(cls.name(), cls)

    def list(self) -> List[str]:
        return list(self._plugins.keys())

    def list_available(self) -> List[str]:
        return list(self._available_plugins.keys())

    def types(self) -> List[Type]:
        return list(self._plugins.values())

    def add(self, name: str, cls: Type) -> "PluginLoader":
        if name in self._plugins:
            raise ValueError(f"Plugin {name} already exists.")

        if not issubclass(cls, self._expected_type):
            raise ValueError(
                f"Plugin {name} must derive the {self._expected_type} interface."
            )

        self._plugins[name] = cls

        return self

    def get(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if name not in self._plugins and name not in self._available_plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        if name not in self._plugins:
            self._load_single_plugin(self._available_plugins[name])

        if name not in self._plugins:
            raise ValueError(f"Plugin {name} cannot be loaded.")

        return self._plugins[name](*args, **kwargs)

    def get_type(self, name: str) -> Type:
        if name not in self._plugins and name not in self._available_plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        if name not in self._plugins:
            self._load_single_plugin(self._available_plugins[name])

        if name not in self._plugins:
            raise ValueError(f"Plugin {name} doesn't exist.")

        return self._plugins[name]

    def __iter__(self) -> Generator:
        for x in self._plugins:
            yield x

    def __len__(self) -> int:
        return len(self.list())

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def reload(self) -> "PluginLoader":
        self._plugins = {}
        return self
