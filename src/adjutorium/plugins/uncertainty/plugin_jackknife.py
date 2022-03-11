# stdlib
import copy
from typing import Any

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# adjutorium absolute
from adjutorium.plugins.uncertainty.base import UncertaintyPlugin

percentile_val = 1.96


class JackknifePlugin(UncertaintyPlugin):
    """
    Uncertainty plugin based on the JackKnife-CV method.

    Args:
        estimator: model. The model to explain.
        X: dataframe. Training set
        y: dataframe. Training labels
        task_type: str. classification or risk_estimation
        prefit: bool. If true, the estimator won't be trained.
        n_epoch: int. training epochs
        subsample: int. Number of samples to use.
        time_to_event: dataframe. Used for risk estimation tasks.
        eval_times: list. Used for risk estimation tasks.
    """

    def __init__(
        self,
        estimator: Any,
        n_folds: int = 3,
        random_seed: int = 0,
    ) -> None:
        if n_folds < 2:
            raise RuntimeError("Please provide at least n_folds >= 2")

        self.estimator = copy.deepcopy(estimator)
        self.n_folds = n_folds
        self.random_seed = random_seed

        self.models: list = []

    def fit(self, *args: Any, **kwargs: Any) -> "UncertaintyPlugin":
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)

        for train_index, _ in kf.split(*args):
            fold_args = []
            for arg in args:
                fold_args.append(arg[train_index])

            fold_model = copy.deepcopy(self.estimator)
            self.models.append(fold_model.fit(*fold_args))

        return self

    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        predictions = []

        for model in self.models:
            predictions.append(model.predict(*args, *kwargs))

        predictions_np = np.asarray(predictions)

        mean = predictions_np.mean(axis=0)
        std = percentile_val * predictions_np.std(axis=0) / np.sqrt(len(predictions))

        return mean.squeeze(), std.squeeze()

    def predict_proba(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        predictions = []

        for model in self.models:
            predictions.append(model.predict_proba(*args, *kwargs))

        predictions_np = np.asarray(predictions)

        mean = predictions_np.mean(axis=0)
        std = percentile_val * predictions_np.std(axis=0) / np.sqrt(len(predictions))

        return mean.squeeze(), std.squeeze()

    @staticmethod
    def name() -> str:
        return "jackknife"


plugin = JackknifePlugin
