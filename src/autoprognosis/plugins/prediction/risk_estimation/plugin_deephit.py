# stdlib
from typing import Any, List, Optional, Tuple

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.prediction.risk_estimation.base as base
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.pip import install
import autoprognosis.utils.serialization as serialization

for retry in range(2):
    try:
        # third party
        from pycox.models import DeepHitSingle
        import torch
        import torchtuples as tt

        break
    except ImportError:
        depends = ["torch", "pycox", "torchtuples"]
        install(depends)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepHitRiskEstimationPlugin(base.RiskEstimationPlugin):
    """DeepHit plugin for survival analysis. DeepHit, that uses a deep neural network to learn the distribution of survival times directly.DeepHit makes no assumptions about the underlying stochastic process and allows for the possibility that the relationship between covariates and risk(s) changes over time. Most importantly, DeepHit smoothly handles competing risks; i.e. settings in which there is more than one possible event of interest.

    Args:
        num_durations: int
            Number of points in the survival function
        batch_size: int
            Batch size
        epochs: int
            Number of iterations
        lr: float
            learning rate
        dim_hidden: int
            Number of neurons in the hidden layers
        alpha: float
            Weighting (0, 1) likelihood and rank loss (L2 in paper). 1 gives only likelihood, and 0 gives only rank loss. (default: {0.2})
        sigma: float
            From eta in rank loss (L2 in paper) (default: {0.1})
        dropout: float
            Dropout value
        patience: int
            Number of epochs without improvement.
        batch_norm: bool
            Enable/Disable batch_norm
        random_state: int
            Random seed

    Example:
        >>> from autoprognosis.plugins.prediction import Predictions
        >>> from pycox.datasets import metabric
        >>>
        >>> df = metabric.read_df()
        >>> X = df.drop(["duration", "event"], axis=1)
        >>> Y = df["event"]
        >>> T = df["duration"]
        >>>
        >>> plugin = Predictions(category="risk_estimation").get("deephit")
        >>> plugin.fit(X, T, Y)
        >>>
        >>> eval_time_horizons = [int(T[Y.iloc[:] == 1].quantile(0.50))]
        >>> plugin.predict(X, eval_time_horizons)

    References:
    [1] Changhee Lee, William R Zame, Jinsung Yoon, and Mihaela van der Schaar. Deephit: A deep learning
        approach to survival analysis with competing risks. In Thirty-Second AAAI Conference on Artificial
        Intelligence, 2018.
        http://medianetlab.ee.ucla.edu/papers/AAAI_2018_DeepHit
    """

    def __init__(
        self,
        model: Any = None,
        num_durations: int = 10,
        batch_size: int = 100,
        epochs: int = 5000,
        lr: float = 1e-2,
        dim_hidden: int = 300,
        alpha: float = 0.28,
        sigma: float = 0.38,
        dropout: float = 0.2,
        patience: int = 20,
        batch_norm: bool = False,
        random_state: int = 0,
        hyperparam_search_iterations: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        enable_reproducible_results(random_state)
        if model:
            self.model = model
            return

        if hyperparam_search_iterations:
            epochs = 10 * int(hyperparam_search_iterations)

        self.model = None
        self.num_durations = num_durations
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.dim_hidden = dim_hidden
        self.alpha = alpha
        self.sigma = sigma
        self.patience = patience
        self.dropout = dropout
        self.batch_norm = batch_norm

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "DeepHitRiskEstimationPlugin":
        if len(args) < 2:
            raise ValueError("Invalid input for fit. Expecting X, T and Y.")

        T = args[0]
        E = args[1]

        labtrans = DeepHitSingle.label_transform(self.num_durations)

        X = np.asarray(X).astype("float32")

        X_train, X_val, E_train, E_val, T_train, T_val = train_test_split(
            X, E, T, random_state=42
        )

        def get_target(df: Any) -> Tuple:
            return (np.asarray(df[0]), np.asarray(df[1]))

        y_train = labtrans.fit_transform(*get_target((T_train, E_train)))
        y_val = labtrans.transform(*get_target((T_val, E_val)))

        in_features = X_train.shape[1]
        out_features = labtrans.out_features

        net = torch.nn.Sequential(
            torch.nn.Linear(in_features, self.dim_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.dim_hidden, self.dim_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.dim_hidden, self.dim_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(self.dim_hidden, out_features),
        ).to(DEVICE)

        self.model = DeepHitSingle(
            net,
            tt.optim.Adam,
            alpha=self.alpha,
            sigma=self.sigma,
            duration_index=labtrans.cuts,
        )

        self.model.optimizer.set_lr(self.lr)

        callbacks = [tt.callbacks.EarlyStopping(patience=self.patience)]
        self.model.fit(
            X_train,
            y_train,
            self.batch_size,
            self.epochs,
            callbacks,
            val_data=(X_val, y_val),
            verbose=False,
        )

        return self

    def _find_nearest(self, array: np.ndarray, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        if len(args) < 1:
            raise ValueError("Invalid input for predict. Expecting X and time horizon.")

        self.model.net.eval()

        time_horizons = args[0]

        X = np.asarray(X).astype("float32")
        surv = self.model.predict_surv_df(X).T

        preds_ = np.zeros([np.shape(surv)[0], len(time_horizons)])

        time_bins = surv.columns
        for t, eval_time in enumerate(time_horizons):
            nearest = self._find_nearest(time_bins, eval_time)
            preds_[:, t] = np.asarray(1 - surv[nearest])

        return preds_

    @staticmethod
    def name() -> str:
        return "deephit"

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return [
            params.Categorical("batch_size", [100, 200, 500]),
            params.Categorical("lr", [1e-2, 1e-3, 1e-4]),
            params.Integer("dim_hidden", 10, 100, 10),
            params.Float("alpha", 0.0, 0.5),
            params.Float("sigma", 0.0, 0.5),
            params.Float("dropout", 0.0, 0.2),
            params.Integer("patience", 10, 50),
        ]

    def save(self) -> bytes:
        return serialization.save_model(self.model)

    @classmethod
    def load(cls, buff: bytes) -> "DeepHitRiskEstimationPlugin":
        model = serialization.load_model(buff)
        return cls(model=model)


plugin = DeepHitRiskEstimationPlugin
