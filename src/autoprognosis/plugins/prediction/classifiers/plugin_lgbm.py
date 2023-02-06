# stdlib
from typing import Any, List

# third party
import pandas as pd

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
from autoprognosis.utils.pip import install
import autoprognosis.utils.serialization as serialization

for retry in range(2):
    try:
        # third party
        import lightgbm as lgbm

        break
    except ImportError:
        depends = ["lightgbm"]
        install(depends)


class LightGBMPlugin(base.ClassifierPlugin):
    """Classification plugin based on LightGBM.

    Method:
        Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. When a decision tree is the weak learner, the resulting algorithm is called gradient boosted trees, which usually outperforms random forest.

    Args:
        n_estimators: int
            The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        learning_rate: float
            Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
        max_depth: int
            The maximum depth of the individual regression estimators.
        boosting_type: str
            ‘gbdt’, traditional Gradient Boosting Decision Tree. ‘dart’, Dropouts meet Multiple Additive Regression Trees. ‘goss’, Gradient-based One-Side Sampling. ‘rf’, Random Forest.
        objective:str
             Specify the learning task and the corresponding learning objective or a custom objective function to be used.
        reg_lambda:float
             L2 regularization term on weights.
        reg_alpha:float
             L1 regularization term on weights.
        colsample_bytree:float
            Subsample ratio of columns when constructing each tree.
        subsample:float
            Subsample ratio of the training instance.
        num_leaves:int
             Maximum tree leaves for base learners.
        min_child_samples:int
            Minimum sum of instance weight (hessian) needed in a child (leaf).
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("lgbm")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        n_estimators: int = 100,
        boosting_type: str = "gbdt",
        learning_rate: float = 1e-2,
        max_depth: int = 6,
        reg_lambda: float = 1e-3,
        reg_alpha: float = 1e-3,
        colsample_bytree: float = 0.1,
        subsample: float = 0.1,
        num_leaves: int = 31,
        min_child_samples: int = 1,
        calibration: int = 0,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        self.boosting_type = boosting_type

        model = lgbm.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            colsample_bytree=self.colsample_bytree,
            subsample=self.subsample,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            boosting_type=self.boosting_type,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "lgbm"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("boosting_type", ["gbdt", "dart", "goss"]),
            params.Integer("num_leaves", 10, 256),
            params.Float("learning_rate", 0.01, 0.3),
            params.Integer("n_estimators", 10, 3000),
            params.Integer("max_depth", 1, 7),
            params.Integer("min_child_samples", 1, 500),
            params.Float("subsample", 0.1, 1.0),
            params.Float("colsample_bytree", 0.1, 1.0),
            params.Float("reg_lambda", 1e-3, 1),
            params.Float("reg_alpha", 1e-3, 1),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "LightGBMPlugin":
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
    def load(cls, buff: bytes) -> "LightGBMPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = LightGBMPlugin
