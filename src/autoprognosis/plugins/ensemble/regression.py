# stdlib
from abc import ABCMeta, abstractmethod
import copy
from typing import Any, Dict, List, Optional

# third party
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

# autoprognosis absolute
import autoprognosis.logger as log
from autoprognosis.plugins.explainers import Explainers
from autoprognosis.plugins.pipeline import PipelineMeta
from autoprognosis.utils.parallel import n_opt_jobs
import autoprognosis.utils.serialization as serialization

dispatcher = Parallel(max_nbytes=None, backend="loky", n_jobs=n_opt_jobs())


class BaseRegressionEnsemble(metaclass=ABCMeta):
    """
    Abstract ensemble interface
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "BaseRegressionEnsemble":
        ...

    @abstractmethod
    def explain(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        ...

    def enable_explainer(
        self,
        explainer_plugins: list = [],
        explanations_nepoch: int = 10000,
    ) -> None:
        self.explainer_plugins = explainer_plugins
        self.explanations_nepoch = explanations_nepoch

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def save(self) -> bytes:
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        ...

    @classmethod
    @abstractmethod
    def load(cls, buff: bytes) -> "BaseRegressionEnsemble":
        ...

    @abstractmethod
    def is_fitted(self) -> bool:
        ...


class WeightedRegressionEnsemble(BaseRegressionEnsemble):
    """
    Weighted ensemble

    Args:
        models: list. List of base models.
        weights: list. The weights for each base model.
        explainer_plugins: list. List of explainers attached to the ensemble.
    """

    def __init__(
        self,
        models: List[PipelineMeta],
        weights: List[float],
        explainer_plugins: list = [],
        explainers: Optional[Dict] = None,
        explanations_nepoch: int = 10000,
    ) -> None:
        super().__init__()

        self.models = []
        self.weights = []
        self.explainer_plugins = explainer_plugins
        self.explanations_nepoch = explanations_nepoch
        self.explainers = explainers

        for idx, weight in enumerate(weights):
            if weight == 0:
                continue
            self.models.append(models[idx])
            self.weights.append(weights[idx])

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "WeightedRegressionEnsemble":
        def fit_model(k: int) -> Any:
            return self.models[k].fit(X, Y)

        log.debug("Fitting the WeightedRegressionEnsemble")
        self.models = dispatcher(delayed(fit_model)(k) for k in range(len(self.models)))

        if self.explainers:
            return self

        self.explainers = {}

        for exp in self.explainer_plugins:
            log.debug("Fitting the explainer for the WeightedRegressionEnsemble")
            exp_model = Explainers().get(
                exp,
                copy.deepcopy(self),
                X,
                Y,
                n_epoch=self.explanations_nepoch,
                prefit=True,
            )
            self.explainers[exp] = exp_model

        return self

    def is_fitted(self) -> bool:
        _fitted = True
        for model in self.models:
            _fitted = _fitted and model.is_fitted()

        return _fitted

    def predict(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        preds_ = []
        for k in range(len(self.models)):
            preds_.append(self.models[k].predict(X, *args) * self.weights[k])
        pred_ens = np.sum(np.array(preds_), axis=0)
        pred_ens = np.asarray(pred_ens)
        if len(pred_ens.shape) < 2:
            pred_ens = pred_ens.reshape(-1, 1)

        return pd.DataFrame(pred_ens)

    def explain(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        if self.explainers is None:
            raise ValueError("Interpretability is not enabled for this ensemble")

        results = {}
        for exp in self.explainers:
            results[exp] = self.explainers[exp].explain(X)

        return results

    def name(self) -> str:
        ensemble_name = []
        for model, weight in zip(self.models, self.weights):
            if weight == 0:
                continue
            ensemble_name.append(f"{weight} * ({model.name()})")

        return " + ".join(ensemble_name)

    def save(self) -> bytes:
        return serialization.save(
            {
                "models": self.models,
                "weights": self.weights,
                "explainers": self.explainers,
            }
        )

    @classmethod
    def load(cls, buff: bytes) -> "WeightedRegressionEnsemble":
        obj = serialization.load(buff)
        return cls(obj["models"], obj["weights"], explainers=obj["explainers"])
