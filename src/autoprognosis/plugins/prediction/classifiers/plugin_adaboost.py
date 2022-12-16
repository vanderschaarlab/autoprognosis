# stdlib
import copy
from typing import Any, List

# third party
from packaging import version
import pandas as pd
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
from autoprognosis.utils.pip import install
import autoprognosis.utils.serialization as serialization

from sklearn.ensemble import HistGradientBoostingClassifier  # isort:skip

for retry in range(2):
    try:
        # third party
        from catboost import CatBoostClassifier
        import lightgbm as lgbm

        break
    except ImportError:
        depends = ["catboost", "lightgbm"]
        install(depends)


class AdaBoostPlugin(base.ClassifierPlugin):
    """Classification plugin based on the AdaBoost estimator.

    Method:
        An AdaBoost classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases.

    Args:
        estimator: int
            Base Learner to use. 0: HistGradientBoostingClassifier, 1: CatBoostClassifier, 2: LGBM, 3: LogisticRegression
        n_estimators: int
            The maximum number of estimators at which boosting is terminated.
        learning_rate: float
            Weight applied to each classifier at each boosting iteration. A higher learning rate increases the contribution of each classifier. There is a trade-off between the learning_rate and n_estimators parameters.
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
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
        estimator: int = 0,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        calibration: int = 0,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if model is not None:
            self.model = model
            return

        if version.parse(sklearn.__version__) >= version.parse("1.2"):
            est_kargs = {
                "estimator": copy.deepcopy(AdaBoostPlugin.base_estimators[estimator]),
            }
        else:
            est_kargs = {
                "base_estimator": copy.deepcopy(
                    AdaBoostPlugin.base_estimators[estimator]
                ),
            }
        model = AdaBoostClassifier(
            **est_kargs,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "adaboost"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("n_estimators", 10, 100, 10),
            params.Categorical("learning_rate", [10**-p for p in range(1, 5)]),
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
