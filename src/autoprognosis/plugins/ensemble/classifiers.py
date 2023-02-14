# stdlib
from abc import ABCMeta, abstractmethod
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

# third party
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold

# autoprognosis absolute
import autoprognosis.logger as log
from autoprognosis.plugins.ensemble.combos import SimpleClassifierAggregator, Stacking
from autoprognosis.plugins.explainers import Explainers
from autoprognosis.plugins.imputers import Imputers
from autoprognosis.plugins.pipeline import Pipeline, PipelineMeta
from autoprognosis.plugins.prediction.classifiers import Classifiers
from autoprognosis.utils.parallel import n_opt_jobs
import autoprognosis.utils.serialization as serialization
from autoprognosis.utils.tester import classifier_metrics

dispatcher = Parallel(max_nbytes=None, backend="loky", n_jobs=n_opt_jobs())


class BaseEnsemble(BaseEstimator, metaclass=ABCMeta):
    """
    Abstract ensemble interface
    """

    def __init__(self) -> None:
        BaseEstimator.__init__(self)

    @abstractmethod
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "BaseEnsemble":
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        ...

    def predict(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        preds = self.predict_proba(X, *args)
        res = np.argmax(preds.values, axis=1)

        return pd.Series(res)

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

    def score(self, X: pd.DataFrame, y: pd.DataFrame, metric: str = "aucroc") -> float:
        ev = classifier_metrics()
        preds = self.predict_proba(X)
        return ev.score_proba(y, preds)[metric]

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def save(self) -> bytes:
        ...

    @classmethod
    @abstractmethod
    def load(cls, buff: bytes) -> "BaseEnsemble":
        ...

    @abstractmethod
    def is_fitted(self) -> bool:
        ...


class WeightedEnsemble(BaseEnsemble):
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

    def is_fitted(self) -> bool:
        _fitted = True
        for model in self.models:
            _fitted = _fitted and model.is_fitted()

        return _fitted

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "WeightedEnsemble":
        def fit_model(k: int) -> Any:
            return self.models[k].fit(X, Y)

        log.debug("Fitting the WeightedEnsemble")
        self.models = dispatcher(delayed(fit_model)(k) for k in range(len(self.models)))

        if self.explainers:
            return self

        self.explainers = {}

        for exp in self.explainer_plugins:
            log.debug("Fitting the explainer for the WeightedEnsemble")
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

    def predict_proba(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        if not self.is_fitted():
            raise RuntimeError("Fit the model first")

        preds_ = []
        for k in range(len(self.models)):
            preds_.append(self.models[k].predict_proba(X, *args) * self.weights[k])
        pred_ens = np.sum(np.array(preds_), axis=0)

        return pd.DataFrame(pred_ens)

    def explain(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        if self.explainers is None:
            raise ValueError("Interpretability is not enabled for this ensemble")

        results = {}
        for exp in self.explainers:
            try:
                results[exp] = self.explainers[exp].explain(X)
            except BaseException as e:
                log.error(f"explainer {exp} failed: {e} ")

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
    def load(cls, buff: bytes) -> "WeightedEnsemble":
        obj = serialization.load(buff)
        return cls(obj["models"], obj["weights"], explainers=obj["explainers"])


class WeightedEnsembleCV(BaseEnsemble):
    """
    Cross-validated Weighted ensemble, with uncertainity prediction support

    Args:
        models: list. List of base models.
        weights: list. The weights for each base model.
        explainer_plugins: list. List of explainers attached to the ensemble.
    """

    def __init__(
        self,
        ensembles: Optional[List[WeightedEnsemble]] = None,
        models: Optional[List[PipelineMeta]] = None,
        weights: Optional[List[float]] = None,
        n_folds: int = 5,
        explainer_plugins: list = [],
        explainers: Optional[dict] = None,
        explanations_nepoch: int = 10000,
    ) -> None:
        super().__init__()

        if ensembles is None and models is None:
            raise ValueError(
                "Invalid input for WeightedEnsembleCV. Provide trained ensembles or raw models."
            )

        if n_folds < 2:
            raise ValueError("Invalid value for n_folds. Must be >= 2.")

        if ensembles is not None:
            self.models = ensembles
        else:
            self.models = []
            if models is None or weights is None:
                raise ValueError(
                    "Invalid input for WeightedEnsemble. Provide the models and the weights."
                )
            for folds in range(n_folds):
                self.models.append(
                    WeightedEnsemble(models, weights, explainer_plugins=[])
                )

        self.explainer_plugins = explainer_plugins
        self.explainers = explainers
        self.explanations_nepoch = explanations_nepoch

        self.n_folds = n_folds
        self.seed = 42

    def is_fitted(self) -> bool:
        _fitted = True
        for model in self.models:
            _fitted = _fitted and model.is_fitted()

        return _fitted

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "WeightedEnsembleCV":
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )
        cv_idx = 0
        for train_index, test_index in skf.split(X, Y):
            X_train = X.loc[X.index[train_index]]
            Y_train = Y.loc[Y.index[train_index]]

            self.models[cv_idx].fit(X_train, Y_train)
            cv_idx += 1

        if self.explainers:
            return self

        self.explainers = {}

        for exp in self.explainer_plugins:
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

    def predict_proba(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        result, _ = self.predict_proba_with_uncertainity(X)

        return result

    def predict_proba_with_uncertainity(
        self, X: pd.DataFrame, *args: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        results = []

        for model in self.models:
            results.append(np.asarray(model.predict_proba(X)))

        results = np.asarray(results)
        calibrated_result = np.mean(results, axis=0)
        uncertainity = 1.96 * np.std(results, axis=0) / np.sqrt(len(results))

        return pd.DataFrame(calibrated_result), pd.DataFrame(uncertainity[:, 0])

    def explain(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        if self.explainers is None:
            raise ValueError("Interpretability is not enabled for this ensemble")

        results = {}
        for exp in self.explainers:
            results[exp] = self.explainers[exp].explain(X)
        return results

    def name(self) -> str:
        return "Calibrated " + self.models[0].name()

    def save(self) -> bytes:
        return serialization.save(
            {
                "ensembles": self.models,
                "explainers": self.explainers,
            }
        )

    @classmethod
    def load(cls, buff: bytes) -> "WeightedEnsembleCV":
        obj = serialization.load(buff)
        return cls(ensembles=obj["ensembles"], explainers=obj["explainers"])


class StackingEnsemble(BaseEnsemble):
    """
    Stacking ensemble(meta ensembling): Use a meta-learner on top of the base models

    Args:
        models: list. List of base models.
        meta_model: Pipeline. The meta learner.
        explainer_plugins: list. List of explainers attached to the ensemble.
    """

    def __init__(
        self,
        models: List[PipelineMeta],
        meta_model: Optional[PipelineMeta] = None,
        clf: Union[None, Stacking] = None,
        explainer_plugins: list = [],
        explanations_nepoch: int = 10000,
    ) -> None:
        super().__init__()

        self.models = models
        if meta_model is None:
            meta_model = Pipeline(
                [
                    Imputers().get_type("ice").fqdn(),
                    Classifiers().get_type("logistic_regression").fqdn(),
                ]
            )()
        meta_model = copy.deepcopy(meta_model)
        meta_model.change_output("numpy")

        self.meta_model = meta_model

        self.explainer_plugins = explainer_plugins
        self.explainers: Optional[dict]
        self.explanations_nepoch = explanations_nepoch

        for model in self.models:
            model.change_output("numpy")

        if clf:
            self.clf = clf
        else:
            self.clf = Stacking(
                models,
                meta_clf=meta_model,
                use_proba=False,
            )

    def is_fitted(self) -> bool:
        _fitted = True
        for model in self.models:
            _fitted = _fitted and model.is_fitted()

        return _fitted

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "StackingEnsemble":
        self.clf.fit(X, Y)

        self.explainers = {}

        for exp in self.explainer_plugins:
            self.explainers[exp] = Explainers().get(
                exp,
                copy.deepcopy(self),
                X,
                Y,
                n_epoch=self.explanations_nepoch,
                prefit=True,
            )
        return self

    def predict_proba(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        if not self.is_fitted():
            raise RuntimeError("Fit the model first")

        return pd.DataFrame(self.clf.predict_proba(X))

    def explain(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        if self.explainers is None:
            raise ValueError("Interpretability is not enabled for this ensemble")

        results = {}

        for exp in self.explainers:
            results[exp] = self.explainers[exp].explain(X)
        return results

    def name(self) -> str:
        ensemble_name = []
        for model in self.models:
            ensemble_name.append(model.name())

        return " + ".join(ensemble_name) + " --> " + self.meta_model.name()

    def save(self) -> bytes:
        return serialization.save(
            {
                "models": self.models,
                "meta_model": self.meta_model,
                "clf": self.clf,
            }
        )

    @classmethod
    def load(cls, buff: bytes) -> "StackingEnsemble":
        obj = serialization.load(buff)
        return cls(obj["models"], obj["meta_model"], clf=obj["clf"])


class AggregatingEnsemble(BaseEnsemble):
    """
    Basic ensemble strategies:
        - average:  average across all scores/prediction results, maybe with weights
        - maximization: simple combination by taking the maximum scores
        - majority vote
        - median: take the median value across all scores/prediction results

    Args:
        models: list. List of base models.
        method: str. average, maximization, majority vote, median
        explainer_plugins: list. List of explainers attached to the ensemble.
    """

    def __init__(
        self,
        models: List[PipelineMeta],
        method: str = "average",
        clf: Union[SimpleClassifierAggregator, None] = None,
        explainer_plugins: list = [],
        explanations_nepoch: int = 10000,
    ) -> None:
        super().__init__()

        self.models = models
        for model in self.models:
            model.change_output("numpy")

        if method not in [
            "average",
            "maximization",
            "majority vote",
            "median",
        ]:
            raise RuntimeError("Invalid aggregating ensemble method")

        self.method = method
        self.explainer_plugins = explainer_plugins
        self.explainers: Optional[dict]
        self.explanations_nepoch = explanations_nepoch

        if clf:
            self.clf = clf
        else:
            self.clf = SimpleClassifierAggregator(models, method=method)

    def is_fitted(self) -> bool:
        _fitted = True
        for model in self.models:
            _fitted = _fitted and model.is_fitted()

        return _fitted

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> "AggregatingEnsemble":
        Y = pd.DataFrame(Y).values.ravel()

        self.clf.fit(X, Y)

        self.explainers = {}

        for exp in self.explainer_plugins:
            self.explainers[exp] = Explainers().get(
                exp,
                copy.deepcopy(self),
                X,
                Y,
                n_epoch=self.explanations_nepoch,
                prefit=True,
            )
        return self

    def predict_proba(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        if not self.is_fitted():
            raise RuntimeError("Fit the model first")

        return pd.DataFrame(self.clf.predict_proba(X))

    def explain(self, X: pd.DataFrame, *args: Any) -> pd.DataFrame:
        if self.explainers is None:
            raise ValueError("Interpretability is not enabled for this ensemble")

        results = {}
        for exp in self.explainers:
            results[exp] = self.explainers[exp].explain(X)

        return results

    def name(self) -> str:
        ensemble_name = []
        for model in self.models:
            ensemble_name.append(model.name())

        return self.method + "( " + " + ".join(ensemble_name) + ")"

    def save(self) -> bytes:
        return serialization.save(
            {
                "models": self.models,
                "method": self.method,
                "clf": self.clf,
            }
        )

    @classmethod
    def load(cls, buff: bytes) -> "AggregatingEnsemble":
        obj = serialization.load(buff)
        return cls(obj["models"], obj["method"], clf=obj["clf"])
