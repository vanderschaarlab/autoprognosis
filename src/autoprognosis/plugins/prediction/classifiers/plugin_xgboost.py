# stdlib
from typing import Any, List, Optional

# third party
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
from autoprognosis.utils.parallel import n_learner_jobs
from autoprognosis.utils.pip import install
import autoprognosis.utils.serialization as serialization

for retry in range(2):
    try:
        # third party
        from xgboost import XGBClassifier

        break
    except ImportError:
        depends = ["xgboost"]
        install(depends)


class XGBoostPlugin(base.ClassifierPlugin):
    """Classification plugin based on the XGBoost classifier.

    Method:
        Gradient boosting is a supervised learning algorithm that attempts to accurately predict a target variable by combining an ensemble of estimates from a set of simpler and weaker models. The XGBoost algorithm has a robust handling of a variety of data types, relationships, distributions, and the variety of hyperparameters that you can fine-tune.

    Args:
        n_estimators: int
            The maximum number of estimators at which boosting is terminated.
        max_depth: int
            Maximum depth of a tree.
        reg_lambda: float
            L2 regularization term on weights (xgb’s lambda).
        reg_alpha: float
            L1 regularization term on weights (xgb’s alpha).
        colsample_bytree: float
            Subsample ratio of columns when constructing each tree.
        colsample_bynode: float
             Subsample ratio of columns for each split.
        colsample_bylevel: float
             Subsample ratio of columns for each level.
        subsample: float
            Subsample ratio of the training instance.
        learning_rate: float
            Boosting learning rate
        booster: int index
            Specify which booster to use: gbtree, gblinear or dart.
        min_child_weight: int
            Minimum sum of instance weight(hessian) needed in a child.
        max_bin: int
            Number of bins for histogram construction.
        grow_policy: int index
            Controls a way new nodes are added to the tree. 0: "depthwise", 1 : "lossguide"
        random_state: float
            Random number seed.
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.

    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("xgboost", n_estimators = 20)
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y)
    """

    booster = ["gbtree", "gblinear", "dart"]
    grow_policy = ["depthwise", "lossguide"]

    def __init__(
        self,
        n_estimators: int = 100,
        reg_lambda: float = 1e-3,
        reg_alpha: float = 1e-3,
        colsample_bytree: float = 0.1,
        colsample_bynode: float = 0.1,
        colsample_bylevel: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.1,
        learning_rate: float = 1e-2,
        min_child_weight: int = 0,
        max_bin: int = 256,
        booster: int = 0,
        grow_policy: int = 0,
        random_state: int = 0,
        calibration: int = 0,
        gamma: float = 0,
        model: Any = None,
        nthread: int = n_learner_jobs(),
        hyperparam_search_iterations: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        if hyperparam_search_iterations:
            n_estimators = int(hyperparam_search_iterations)

        model = XGBClassifier(
            n_estimators=n_estimators,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
            colsample_bylevel=colsample_bylevel,
            max_depth=max_depth,
            subsample=subsample,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            max_bin=max_bin,
            verbosity=0,
            booster=XGBoostPlugin.booster[booster],
            grow_policy=XGBoostPlugin.grow_policy[grow_policy],
            random_state=random_state,
            nthread=nthread,
            gamma=gamma,
            **kwargs,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "xgboost"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("max_depth", 1, 7),
            params.Float("learning_rate", 1e-3, 0.3),
            params.Integer("n_estimators", 10, 10000),
            params.Float("colsample_bytree", 0.1, 0.5),
            params.Float("gamma", 0, 1),
            params.Float("subsample", 0.5, 1),
            params.Float("reg_lambda", 1e-3, 10.0),
            params.Float("reg_alpha", 1e-3, 10.0),
            params.Float("colsample_bynode", 0.1, 0.9),
            params.Float("colsample_bylevel", 0.1, 0.9),
            params.Integer("min_child_weight", 0, 300),
            params.Integer("max_bin", 256, 512),
            params.Integer("grow_policy", 0, len(XGBoostPlugin.grow_policy) - 1),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "XGBoostPlugin":
        y = np.asarray(args[0])

        self.encoder = LabelEncoder()
        y = self.encoder.fit_transform(y)

        classes_weights = class_weight.compute_sample_weight(
            class_weight="balanced", y=y
        )
        self.model.fit(X, y, sample_weight=classes_weights, **kwargs)
        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return self.encoder.inverse_transform(self.model.predict(X, *args, **kwargs))

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        return self.model.predict_proba(X, *args, **kwargs)

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "XGBoostPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = XGBoostPlugin
