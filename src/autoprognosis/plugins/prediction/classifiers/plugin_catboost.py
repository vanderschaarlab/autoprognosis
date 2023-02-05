# stdlib
from typing import Any, List, Optional

# third party
import pandas as pd

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.utils.pip import install
import autoprognosis.utils.serialization as serialization

for retry in range(2):
    try:
        # third party
        from catboost import CatBoostClassifier

        break
    except ImportError:
        depends = ["catboost"]
        install(depends)


class CatBoostPlugin(base.ClassifierPlugin):
    """Classification plugin based on the CatBoost framework.

    Method:
        CatBoost provides a gradient boosting framework which attempts to solve for Categorical features using a permutation driven alternative compared to the classical algorithm. It uses Ordered Boosting to overcome over fitting and Symmetric Trees for faster execution.

    Args:
        n_estimators: int
            Number of gradient boosted trees. Equivalent to number of boosting rounds.
        depth: int
            Depth of the tree.
        grow_policy: int
            The tree growing policy. Defines how to perform greedy tree construction: [SymmetricTree, Depthwise]
        l2_leaf_reg: float
            Coefficient at the L2 regularization term of the cost function.
        learning_rate: float
            The learning rate used for reducing the gradient step.
        min_data_in_leaf: int
            The minimum number of training samples in a leaf.
        random_strength: float
            The amount of randomness to use for scoring splits when the tree structure is selected. Use this parameter to avoid overfitting the model.
        random_state: int, default 0
            Random seed

    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("catboost")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    grow_policies = ["Depthwise", "SymmetricTree", "Lossguide"]

    def __init__(
        self,
        n_estimators: int = 100,
        depth: int = 5,
        grow_policy: int = 0,
        l2_leaf_reg: float = 3,
        learning_rate: float = 1e-3,
        min_data_in_leaf: int = 1,
        random_strength: float = 1,
        random_state: int = 0,
        model: Any = None,
        hyperparam_search_iterations: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        if hyperparam_search_iterations:
            n_estimators = int(hyperparam_search_iterations)

        self.model = CatBoostClassifier(
            depth=depth,
            logging_level="Silent",
            allow_writing_files=False,
            used_ram_limit="10gb",
            n_estimators=n_estimators,
            random_state=random_state,
            grow_policy=CatBoostPlugin.grow_policies[grow_policy],
            l2_leaf_reg=l2_leaf_reg,
            learning_rate=learning_rate,
            min_data_in_leaf=min_data_in_leaf,
            random_strength=random_strength,
            auto_class_weights="Balanced",
        )

    @staticmethod
    def name() -> str:
        return "catboost"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("depth", 1, 7),
            params.Integer("n_estimators", 10, 10000),
            params.Float("learning_rate", 1e-2, 4e-2),
            params.Integer("grow_policy", 0, len(CatBoostPlugin.grow_policies) - 1),
            params.Float("l2_leaf_reg", 1e-4, 1e3),
            params.Float("random_strength", 0, 3),
            params.Integer("min_data_in_leaf", 1, 300),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "CatBoostPlugin":
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
    def load(cls, buff: bytes) -> "CatBoostPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = CatBoostPlugin
