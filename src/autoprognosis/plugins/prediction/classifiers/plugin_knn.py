# stdlib
from typing import Any, List

# third party
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.classifiers.base as base
from autoprognosis.plugins.prediction.classifiers.helper_calibration import (
    calibrated_model,
)
import autoprognosis.utils.serialization as serialization


class KNNPlugin(base.ClassifierPlugin):
    """Classification plugin based on the k-nearest neighbors vote.

    Method:
        Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

    Args:
        n_neighbors: int
            Number of neighbors to use
        weights: str
            Weight function used in prediction. Possible values: "uniform", "distance"
        algorithm: str
            Algorithm used to compute the nearest neighbors: "ball_tree", "kd_tree", "brute" or "auto".
        leaf_size: int
            Leaf size passed to BallTree or KDTree.
        p: int
            Power parameter for the Minkowski metric.
        calibration: int
            Enable/disable calibration. 0: disabled, 1 : sigmoid, 2: isotonic.
        random_state: int, default 0
            Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="classifiers").get("knn")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    algorithms = ["auto", "ball_tree", "kd_tree", "brute"]
    weights = ["uniform", "distance"]

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: int = 0,
        algorithm: int = 0,
        leaf_size: int = 30,
        p: int = 2,
        calibration: int = 0,
        model: Any = None,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if model is not None:
            self.model = model
            return

        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=KNNPlugin.weights[weights],
            algorithm=KNNPlugin.algorithms[algorithm],
            leaf_size=leaf_size,
            p=p,
        )
        self.model = calibrated_model(model, calibration)

    @staticmethod
    def name() -> str:
        return "knn"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("p", [1, 2]),
            params.Integer("algorithm", 0, len(KNNPlugin.algorithms) - 1),
            params.Integer("weights", 0, len(KNNPlugin.weights) - 1),
            params.Integer("n_neighbors", 1, 50),
        ]

    def _fit(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> "KNNPlugin":
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
    def load(cls, buff: bytes) -> "KNNPlugin":
        model = serialization.load_model(buff)

        return cls(model=model)


plugin = KNNPlugin
