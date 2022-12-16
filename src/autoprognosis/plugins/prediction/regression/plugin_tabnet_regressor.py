# stdlib
from typing import Any, List

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.regression.base as base
from autoprognosis.utils.pip import install
from autoprognosis.utils.serialization import load_model, save_model

for retry in range(2):
    try:
        # third party
        from pytorch_tabnet.tab_model import TabNetRegressor
        import torch

        break
    except ImportError:
        depends = ["torch", "pytorch_tabnet"]
        install(depends)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabNetRegressorPlugin(base.RegressionPlugin):
    """Regression plugin based on TabNet. TabNet uses sequential attention to choose which features to reason from at each decision step, enabling interpretability and more efficient learning as the learning capacity is used for the most salient features.

    Args:
        n_d: int
            Width of the decision prediction layer. Bigger values gives more capacity to the model with the risk of overfitting. Values typically range from 8 to 64.
        n_a: int
            Width of the attention embedding for each mask. According to the paper n_d=n_a is usually a good choice. (default=8)
        lr: float
            Learning rate
        n_steps: int
            Number of steps in the architecture (usually between 3 and 10)
        gamma: float
            This is the coefficient for feature reusage in the masks. A value close to 1 will make mask selection least correlated between layers. Values range from 1.0 to 2.0.
        n_independent: int
            Number of independent Gated Linear Units layers at each step. Usual values range from 1 to 5.
        n_shared: int
            Number of shared Gated Linear Units at each step Usual values range from 1 to 5
        lambda_sparse: float
            This is the extra sparsity loss coefficient as proposed in the original paper. The bigger this coefficient is, the sparser your model will be in terms of feature selection. Depending on the difficulty of your problem, reducing this value could help.

        momentum: float
            Momentum for batch normalization, typically ranges from 0.01 to 0.4 (default=0.02)
        clip_value: float
           If a float is given this will clip the gradient at clip_value.
        max_epochs: int
            Maximum number of epochs for trainng.
        patience: int
            Number of consecutive epochs without improvement before performing early stopping.
        batch_size: int
            Batch size
        random_state: int
            Random seed


    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> plugin = Predictions(category="regression").get("tabnet")
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> plugin.fit_predict(X, y) # returns the probabilities for each class
    """

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        lr: float = 1e-3,
        n_steps: int = 3,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-4,
        momentum: float = 0.3,
        clip_value: float = 2.0,
        epsilon: float = 1e-15,
        n_iter: int = 1000,
        patience: int = 50,
        batch_size: int = 50,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.patience = patience
        self.batch_size = batch_size
        self.max_epochs = n_iter

        self._model = TabNetRegressor(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            lambda_sparse=lambda_sparse,
            momentum=momentum,
            clip_value=clip_value,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=lr),
            scheduler_params={"gamma": 0.95, "step_size": 20},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            epsilon=epsilon,
            verbose=0,
            seed=random_state,
        )

    @staticmethod
    def name() -> str:
        return "tabnet_regressor"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Integer("n_d", 8, 64),
            params.Integer("n_a", 8, 64),
            params.Categorical("lr", [1e-2, 1e-3, 1e-4]),
            params.Integer("n_steps", 3, 10),
            params.Float("gamma", 1.0, 2.0),
            params.Integer("n_independent", 1, 5),
            params.Integer("n_shared", 1, 5),
            params.Float("momentum", 0.01, 0.4),
        ]

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "TabNetRegressorPlugin":
        if len(*args) == 0:
            raise RuntimeError("Please provide the labels for training")

        X = np.asarray(X)
        y = np.asarray(args[0]).reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
        )

        return self

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        X = np.asarray(X)
        return np.asarray(self._model.predict(X))

    def _predict_proba(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        X = np.asarray(X)
        return np.asarray(self._model.predict_proba(X))

    def save(self) -> bytes:
        return save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "TabNetRegressorPlugin":
        return load_model(buff)


plugin = TabNetRegressorPlugin
