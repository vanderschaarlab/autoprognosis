# stdlib
from typing import Any, List

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
        from pycox.models import CoxPH
        import torch
        import torchtuples as tt

        break
    except ImportError:
        depends = ["torch", "pycox", "torchtuples"]
        install(depends)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoxnetRiskEstimationPlugin(base.RiskEstimationPlugin):
    """CoxPH neural net plugin for survival analysis.

    Args:
        hidden_dim: int
            Number of neurons in the hidden layers
        hidden_len: int
            Number of hidden layers
        batch_norm: bool.
            Batch norm on/off.
        dropout: float.
            Dropout value.
        lr: float.
            Learning rate.
        epochs: int.
            Number of training epochs
        patience: int.
            Number of iterations without validation improvement.
        batch_size: int.
            Batch size
        verbose: bool.
            Enable debug logs
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
        >>> plugin = Predictions(category="risk_estimation").get("coxnet")
        >>> plugin.fit(X, T, Y)
        >>>
        >>> eval_time_horizons = [int(T[Y.iloc[:] == 1].quantile(0.50))]
        >>> plugin.predict(X, eval_time_horizons)

    """

    def __init__(
        self,
        hidden_dim: int = 100,
        hidden_len: int = 2,
        batch_norm: bool = True,
        dropout: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 5000,
        patience: int = 50,
        batch_size: int = 128,
        verbose: bool = False,
        random_state: int = 0,
        **kwargs: Any
    ) -> None:

        super().__init__(**kwargs)

        enable_reproducible_results(random_state)

        self.num_nodes = [hidden_dim] * hidden_len

        self.out_features = 1
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.output_bias = False

        self.callbacks = [tt.cb.EarlyStopping(patience=patience)]
        self.epochs = epochs
        self.verbose = verbose
        self.batch_size = batch_size
        self.lr = lr

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        """
        Return the hyperparameter space for the current model.
        """
        return [
            params.Categorical("batch_norm", [1, 0]),
            params.Categorical("dropout", [0, 0.1, 0.2]),
            params.Categorical("lr", [1e-2, 1e-3, 1e-4]),
            params.Integer("patience", 10, 50, 10),
            params.Integer("hidden_dim", 10, 200),
            params.Integer("hidden_len", 1, 4),
        ]

    @staticmethod
    def name() -> str:
        """
        Return the name of the current model.
        """
        return "coxnet"

    def _fit(
        self, X: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> "CoxnetRiskEstimationPlugin":
        if len(args) < 2:
            raise ValueError("Invalid input for fit. Expecting X, T and Y.")

        X = np.asarray(X).astype("float32")
        T = np.asarray(args[0])
        E = np.asarray(args[1])

        X_train, X_val, E_train, E_val, T_train, T_val = train_test_split(
            X, E, T, random_state=0
        )

        y_train = (T_train, E_train)
        y_val = (T_val, E_val)

        self.net = tt.practical.MLPVanilla(
            X.shape[1],
            self.num_nodes,
            self.out_features,
            bool(self.batch_norm),
            self.dropout,
            output_bias=self.output_bias,
        ).to(DEVICE)

        self.model = CoxPH(self.net, tt.optim.Adam)
        self.model.optimizer.set_lr(self.lr)

        self.model.fit(
            X_train,
            y_train,
            self.batch_size,
            self.epochs,
            self.callbacks,
            self.verbose,
            val_data=(X_val, y_val),
            val_batch_size=self.batch_size,
        )
        self.model.compute_baseline_hazards()

        return self

    def _find_nearest(self, array: np.ndarray, value: float) -> float:
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def _predict(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """
        Predict the survival function for the current input.
        """
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

    def save(self) -> bytes:
        return serialization.save_model(self)

    @classmethod
    def load(cls, buff: bytes) -> "CoxnetRiskEstimationPlugin":
        return serialization.load_model(buff)


plugin = CoxnetRiskEstimationPlugin
