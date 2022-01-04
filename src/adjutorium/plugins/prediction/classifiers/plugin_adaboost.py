# stdlib
import copy
from typing import Any, List

# third party
from catboost import CatBoostClassifier
import lightgbm as lgbm
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
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


class AdaBoostPlugin(base.ClassifierPlugin):
    """Classification plugin based on the AdaBoost estimator.

    Method:
        An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

    Args:
        n_estimators: int
            The maximum number of estimators at which boosting is terminated.
        learning_rate: float
            Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier. There is a trade-off between the learning_rate and n_estimators parameters.
        base_estimator: int
            Base estimator to use

    Example:
        >>> from adjutorium.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("adaboost")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
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
    calibrations = ["none", "sigmoid", "isotonic"]

    def __init__(
        self,
        base_estimator: int = 0,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        calibration: int = 0,
        model: Any = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if model is not None:
            self.model = model
            return

        model = AdaBoostClassifier(
            base_estimator=copy.deepcopy(
                AdaBoostPlugin.base_estimators[base_estimator]
            ),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "adaboost"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("n_estimators", 10, 100, 10),
            params.Categorical("learning_rate", [10 ** -p for p in range(1, 5)]),
            params.Integer(
                "base_estimator",
                0,
                len(AdaBoostPlugin.base_estimators) - 1,
            ),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "AdaBoostPlugin":
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
    def load(cls, buff: bytes) -> "AdaBoostPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = AdaBoostPlugin
