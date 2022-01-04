# stdlib
import copy
from typing import Any, List

# third party
from catboost import CatBoostClassifier
import lightgbm as lgbm
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

# adjutorium absolute
import adjutorium.plugins.core.params as params
import adjutorium.plugins.prediction.classifiers.base as base
from adjutorium.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import adjutorium.utils.serialization as serialization

from sklearn.experimental import (  # noqa: F401,E402, isort:skip
    enable_hist_gradient_boosting,
)

from sklearn.ensemble import HistGradientBoostingClassifier  # isort:skip


class BaggingPlugin(base.ClassifierPlugin):
    """Classification plugin based on the Bagging estimator.

    Method:
        A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.

    Args:
        n_estimators: int
            The number of base estimators in the ensemble.
        max_samples: float
            The number of samples to draw from X to train each base estimator.
        max_features: float
            The number of features to draw from X to train each base estimator.
        base_estimator: int
            Base estimator to use

    Example:
        >>> from adjutorium.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("bagging")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    base_estimators = [
        HistGradientBoostingClassifier(max_depth=3),
        CatBoostClassifier(
            max_depth=3,
            logging_level="Silent",
            allow_writing_files=False,
        ),
        lgbm.LGBMClassifier(max_depth=3),
        LogisticRegression(max_iter=10000),
    ]

    def __init__(
        self,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        base_estimator: int = 0,
        calibration: int = 0,
        model: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if model is not None:
            self.model = model
            return

        model = BaggingClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            max_samples=max_samples,
            base_estimator=copy.deepcopy(BaggingPlugin.base_estimators[base_estimator]),
            n_jobs=2,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "bagging"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("n_estimators", 10, 100, 10),
            params.Float("max_samples", 0.01, 1),
            params.Float("max_features", 0.005, 1),
            params.Integer(
                "base_estimator",
                0,
                len(BaggingPlugin.base_estimators) - 1,
            ),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "BaggingPlugin":
        self.model.fit(X, *args, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.model.predict(X, *args, **kwargs)

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "BaggingPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = BaggingPlugin
