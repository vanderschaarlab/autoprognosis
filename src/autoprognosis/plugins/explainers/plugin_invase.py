# stdlib
from abc import ABCMeta, abstractmethod
import copy
import itertools
from typing import Any, Generator, List, Optional, Union

# third party
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample

# autoprognosis absolute
import autoprognosis.logger as log
from autoprognosis.plugins.explainers.base import ExplainerPlugin
from autoprognosis.utils.distributions import enable_reproducible_results
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        import matplotlib.pyplot as plt
        import seaborn as sns
        import torch
        from torch import nn

        break
    except ImportError:
        depends = ["matplotlib", "seaborn", "torch"]
        install(depends)


EPS = 1e-8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample(X: np.ndarray, nsamples: int = 100, random_state: int = 0) -> np.ndarray:
    if nsamples >= X.shape[0]:
        return X
    else:
        return resample(X, n_samples=nsamples, random_state=random_state)


def bitmasks(n: int, m: int) -> Generator:
    if m < n:
        if m > 0:
            for x in bitmasks(n - 1, m - 1):
                yield [1] + x
            for x in bitmasks(n - 1, m):
                yield [0] + x
        else:
            yield [0] * n
    else:
        yield [1] * n


def bitmask_intervals(n: int, low: int, high: int) -> Generator:
    for k in range(low, high):
        for result in bitmasks(n, k):
            yield torch.from_numpy(np.asarray(result))


class Masking(nn.Module):
    def __init__(self, masking_values: torch.Tensor) -> None:
        super(Masking, self).__init__()
        self.masking_values = masking_values

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        if len(tensors) != 2:
            raise RuntimeError(
                "Invalid number of tensor for the masking layer. It requires the features vector and the selection vector"
            )

        features_vector = tensors[0]
        selection_vector = tensors[1]

        if len(features_vector[0]) != len(self.masking_values):
            raise RuntimeError("Invalid shape for the features vector")

        if len(selection_vector[0]) != len(self.masking_values):
            raise RuntimeError("Invalid shape for the features vector")

        sampled_mask = []
        for uniq_vals in self.masking_values:
            rand = np.random.choice(uniq_vals, len(features_vector))
            sampled_mask.append(rand)

        sampled_mask = np.asarray(sampled_mask).T
        sampled_mask = torch.from_numpy(sampled_mask).to(DEVICE)

        sampled_mask[(features_vector == sampled_mask)] = -1

        result = (
            features_vector * selection_vector + (1 - selection_vector) * sampled_mask
        )
        result = result.float()

        return result


class invaseBase(metaclass=ABCMeta):
    def __init__(
        self,
        estimator: Any,
        X: np.ndarray,
        n_epoch: int = 10000,
        n_epoch_inner: int = 1,
        patience: int = 5,
        min_epochs: int = 100,
        n_epoch_print: int = 50,
        batch_size: int = 300,
        learning_rate: float = 1e-3,
        penalty_l2: float = 1e-3,
        feature_names: List = [],
    ) -> None:
        self.batch_size = batch_size  # Batch size
        self.epochs = n_epoch  # Epoch size (large epoch is needed due to the policy gradient framework)
        self.epochs_inner = n_epoch_inner
        self.feature_names = feature_names

        self.patience = patience
        self.min_epochs = min_epochs
        self.n_epoch_print = n_epoch_print
        self.learning_rate = learning_rate
        self.penalty_l2 = penalty_l2

        # Build error predictor
        self.critic = self._build_critic().to(DEVICE)

        self._train(estimator, X)

    @abstractmethod
    def explain(self, X: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        ...

    @abstractmethod
    def _build_critic(self) -> nn.Module:
        ...

    @abstractmethod
    def _baseline_metric(
        self, estimator: Any, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def _baseline_predict(self, estimator: Any, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def _importance_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def _importance_init(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def _importance_test(
        self, estimator: Any, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        ...

    def _train(self, estimator: Any, x: np.ndarray) -> "invaseBase":
        critic_solver = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate,
            weight_decay=self.penalty_l2,
        )

        y = self._baseline_predict(estimator, x)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        x_train = torch.from_numpy(np.asarray(x_train)).float().to(DEVICE)
        y_train = torch.from_numpy(np.asarray(y_train)).float().squeeze().to(DEVICE)

        x_test = torch.from_numpy(np.asarray(x_test)).float().to(DEVICE)
        y_test = torch.from_numpy(np.asarray(y_test)).float().squeeze().to(DEVICE)

        patience = 0
        best_val_loss = 99999999

        n = x_train.shape[0]

        batch_size = self.batch_size if self.batch_size < n else n
        n_batches = int(np.round(n / batch_size)) if batch_size < n else 1
        train_indices = np.arange(n)

        # Train critic NN
        for epoch in range(self.epochs):
            np.random.shuffle(train_indices)
            train_loss = []
            for b in range(n_batches):
                # Select batch
                idx = train_indices[(b * batch_size) : min((b + 1) * batch_size, n - 1)]
                x_batch = x_train[idx, :]
                y_batch = y_train[idx]

                importance = self._importance_test(estimator, x_batch, y_batch).detach()

                critic_solver.zero_grad()

                # Train the critic
                predicted_importance = self.critic(x_batch).float()

                predicted_importance_loss = self._importance_loss(
                    predicted_importance, importance
                )

                predicted_importance_loss.backward()

                critic_solver.step()

                train_loss.append(predicted_importance_loss.detach())

            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if epoch % self.n_epoch_print == 0:
                with torch.no_grad():
                    importance = self._importance_test(estimator, x_test, y_test)
                    predicted_importance = self.critic(x_test)

                    val_loss = self._importance_loss(predicted_importance, importance)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience = 0
                    else:
                        patience += 1
                    if patience > self.patience and epoch > self.min_epochs:
                        break

                log.info(
                    f"Epoch: {epoch}, training invase loss: {torch.mean(train_loss)}  validation loss: {val_loss}"
                )

        return self


class invaseClassifier(invaseBase):
    def __init__(
        self,
        estimator: Any,
        X: np.ndarray,
        critic_latent_dim: int = 200,
        n_epoch: int = 10000,
        n_epoch_inner: int = 2,
        patience: int = 5,
        min_epochs: int = 100,
        n_epoch_print: int = 50,
        batch_size: int = 300,
        learning_rate: float = 1e-3,
        penalty_l2: float = 1e-3,
        feature_names: List = [],
    ) -> None:
        X = np.asarray(X)
        self.latent_dim2 = critic_latent_dim  # Dimension of critic network

        self.input_shape = X.shape[1]  # Input dimension

        masking_values = []
        for col in X.T:
            masking_values.append(np.unique(col))
        self.masking = Masking(masking_values)

        super().__init__(
            estimator=estimator,
            X=X,
            n_epoch=n_epoch,
            n_epoch_inner=n_epoch_inner,
            patience=patience,
            min_epochs=min_epochs,
            n_epoch_print=n_epoch_print,
            batch_size=batch_size,
            learning_rate=learning_rate,
            penalty_l2=penalty_l2,
            feature_names=feature_names,
        )

    def explain(self, X: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        X = np.asarray(X)
        X = torch.from_numpy(X).float().to(DEVICE)

        gen_prob = self.critic(X)

        return gen_prob.detach().cpu().numpy()

    def _build_critic(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_shape, self.latent_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.latent_dim2, self.latent_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.latent_dim2, self.latent_dim2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.latent_dim2, self.input_shape),
            nn.Sigmoid(),
        )

    def _baseline_metric(
        self, estimator: Any, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        df = pd.DataFrame(x.detach().cpu().numpy(), columns=self.feature_names)
        if hasattr(estimator, "predict_proba"):
            baseline_proba = estimator.predict_proba(df)
            baseline_proba = torch.from_numpy(np.asarray(baseline_proba)).to(DEVICE)
            return -torch.sum(y * torch.log(baseline_proba + EPS), dim=-1)
        else:
            baseline_proba = estimator.predict(df)
            baseline_proba = torch.from_numpy(np.asarray(baseline_proba)).to(DEVICE)
            return torch.sum((y - baseline_proba) ** 2, dim=-1)

    def _baseline_predict(self, estimator: Any, x: torch.Tensor) -> torch.Tensor:
        df = pd.DataFrame(x, columns=self.feature_names)
        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(df)
        else:
            return estimator.predict(df)

    def _importance_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return nn.MSELoss()(y_pred, y_true)

    def _importance_init(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape).to(DEVICE)

    def _importance_test(
        self, estimator: Any, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        importance = self._importance_init(x)
        n_features = x.shape[-1]
        # get baseline importance
        for mask in bitmask_intervals(n_features, n_features - 1, n_features):
            mask = torch.broadcast_to(mask, x.shape).to(DEVICE)
            masked_batch = self.masking([x, mask])

            baseline_loss = self._baseline_metric(estimator, masked_batch, y)

            importance = torch.max(importance, ((1 - mask).T * baseline_loss).T)

        # interaction importance
        bitmask_generator = bitmask_intervals(
            n_features, n_features - 3, n_features - 1
        )
        next_slice = list(itertools.islice(bitmask_generator, len(x)))

        while len(next_slice) == len(x):
            next_mask = torch.stack(next_slice).to(DEVICE)

            for local_inter in range(self.epochs_inner):
                indices = torch.argsort(torch.rand(*next_mask.shape), dim=-1)
                mask = next_mask[
                    torch.arange(next_mask.shape[0]).unsqueeze(-1), indices
                ]

                masked_batch = self.masking([x, mask])

                baseline_loss = self._baseline_metric(estimator, masked_batch, y)

                local_importance = 1e-3 * ((1 - mask).T * baseline_loss).T

                importance += local_importance

            next_slice = list(itertools.islice(bitmask_generator, len(x)))

        importance -= importance.min(-1, keepdim=True)[0]
        importance /= importance.max(-1, keepdim=True)[0] + EPS

        return importance.float()


class invaseRiskEstimation(invaseBase):
    def __init__(
        self,
        estimator: Any,
        X: np.ndarray,
        eval_times: List,
        critic_latent_dim: int = 200,
        n_epoch: int = 10000,
        n_epoch_inner: int = 2,
        patience: int = 5,
        min_epochs: int = 100,
        n_epoch_print: int = 10,
        batch_size: int = 500,
        learning_rate: float = 1e-3,
        penalty_l2: float = 1e-3,
        samples: int = 20000,
        feature_names: List = [],
    ) -> None:
        if len(feature_names) == X.shape[1]:
            X = pd.DataFrame(X, columns=feature_names)
        else:
            X = pd.DataFrame(X)
        self.columns = X.columns
        self.eval_times = eval_times

        self.latent_dim2 = critic_latent_dim  # Dimension of critic network

        self.input_shape = X.shape[1]  # Input dimension

        masking_values = []
        for col in X.columns:
            masking_values.append(list(X[col].unique()))
        self.masking = Masking(masking_values)

        X_sampled = sample(X, nsamples=samples)

        super().__init__(
            estimator=estimator,
            X=X_sampled,
            n_epoch=n_epoch,
            n_epoch_inner=n_epoch_inner,
            patience=patience,
            min_epochs=min_epochs,
            n_epoch_print=n_epoch_print,
            batch_size=batch_size,
            learning_rate=learning_rate,
            penalty_l2=penalty_l2,
            feature_names=feature_names,
        )

    def _build_critic(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.input_shape, self.latent_dim2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim2, self.latent_dim2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim2, self.latent_dim2),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim2, self.input_shape * len(self.eval_times)),
        )

    def _baseline_metric(
        self, estimator: Any, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        df = pd.DataFrame(x.detach().cpu().numpy(), columns=self.feature_names)
        baseline_proba = estimator.predict(
            df,
            self.eval_times,
        )
        baseline_proba = torch.from_numpy(np.asarray(baseline_proba)).to(DEVICE)

        out = (baseline_proba - y) ** 2 + torch.abs(baseline_proba - y)
        out += -y * torch.log(baseline_proba + EPS)

        return out

    def _importance_loss(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return nn.MSELoss()(y_pred.view(y_true.shape), y_true)

    def _baseline_predict(self, estimator: Any, x: torch.Tensor) -> torch.Tensor:
        df = pd.DataFrame(x, columns=self.feature_names)
        return estimator.predict(df, self.eval_times)

    def _importance_init(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], x.shape[1], len(self.eval_times))).to(DEVICE)

    def _importance_test(
        self, estimator: Any, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        importance = self._importance_init(x)
        n_features = x.shape[-1]

        # get baseline importance
        for mask in bitmask_intervals(n_features, n_features - 1, n_features):
            mask = torch.broadcast_to(mask, x.shape).to(DEVICE)
            masked_batch = self.masking([x, mask])

            baseline_loss = self._baseline_metric(estimator, masked_batch, y)

            for idx in range(len(self.eval_times)):
                importance[:, :, idx] = torch.max(
                    importance[:, :, idx], ((1 - mask).T * baseline_loss[:, idx]).T
                )

        # interaction importance
        bitmask_generator = bitmask_intervals(
            n_features, n_features - 2, n_features - 1
        )
        next_slice = list(itertools.islice(bitmask_generator, len(x)))

        while len(next_slice) == len(x):
            next_mask = torch.stack(next_slice).to(DEVICE)

            for local_inter in range(self.epochs_inner):
                indices = torch.argsort(torch.rand(*next_mask.shape), dim=-1)
                mask = next_mask[
                    torch.arange(next_mask.shape[0]).unsqueeze(-1), indices
                ]

                masked_batch = self.masking([x, mask])

                baseline_loss = self._baseline_metric(estimator, masked_batch, y)

                for idx in range(len(self.eval_times)):
                    importance[:, :, idx] += (
                        1e-3 * ((1 - mask).T * baseline_loss[:, idx]).T
                    )

            next_slice = list(itertools.islice(bitmask_generator, len(x)))

        # importance = importance.permute(0, 2, 1)
        # importance = (importance - importance.min(-1, keepdim=True)[0]) / (importance.max(-1, keepdim=True)[0] - importance.min(-1, keepdim=True)[0] + EPS)
        # importance = importance.permute(0, 2, 1)

        return importance.float()

    def explain(self, X: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        X = np.asarray(X)
        X = torch.from_numpy(X).float().to(DEVICE)

        gen_prob = self.critic(X).reshape(X.shape[0], X.shape[1], len(self.eval_times))
        return gen_prob.detach().cpu().numpy()


class invaseCV:
    def __init__(
        self,
        estimator: Any,
        X: np.ndarray,
        critic_latent_dim: int = 200,
        n_epoch: int = 10000,
        n_epoch_inner: int = 2,
        patience: int = 5,
        min_epochs: int = 100,
        n_epoch_print: int = 50,
        n_folds: int = 5,
        seed: int = 42,
        feature_names: List = [],
    ) -> None:
        X = np.asarray(X)
        self.feature_names = feature_names

        self.fold_models = []

        skf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_index, test_index in skf.split(X):
            self.fold_models.append(
                invaseClassifier(
                    estimator,
                    X[train_index],
                    critic_latent_dim=critic_latent_dim,
                    n_epoch=n_epoch,
                    n_epoch_inner=n_epoch_inner,
                    patience=patience,
                    min_epochs=min_epochs,
                    feature_names=feature_names,
                )
            )

    def explain(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        result = []
        for fold in self.fold_models:
            fold_out = fold.explain(x)
            result.append(fold_out)
        return np.mean(result, axis=0)


class INVASEPlugin(ExplainerPlugin):
    """
    Interpretability plugin based on the INVASE algorithm.

    Args:
        estimator: model.
            The model to explain.
        X: dataframe.
            Training set
        y: dataframe.
            Training labels
        time_to_event: dataframe.
            Used for risk estimation tasks.
        eval_times: list.
            Used for risk estimation tasks.
        n_epoch: int.
            training epochs
        task_type: str.
            classification or risk_estimation
        samples: int.
            Number of samples to use.
        prefit: bool.
            If true, the estimator won't be trained.

    Example:
        >>> import pandas as pd
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>>from autoprognosis.plugins.explainers import Explainers
        >>> from autoprognosis.plugins.prediction.classifiers import Classifiers
        >>>
        >>> X, y = load_iris(return_X_y=True)
        >>>
        >>> X = pd.DataFrame(X)
        >>> y = pd.Series(y)
        >>>
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> model = Classifiers().get("logistic_regression")
        >>>
        >>> explainer = Explainers().get(
        >>>     "invase",
        >>>     model,
        >>>     X_train,
        >>>     y_train,
        >>>     task_type="classification",
        >>> )
        >>>
        >>> explainer.explain(X_test)
    """

    def __init__(
        self,
        estimator: Any,
        X: pd.DataFrame,
        y: pd.DataFrame,
        time_to_event: Optional[pd.DataFrame] = None,  # for survival analysis
        eval_times: Optional[List] = None,  # for survival analysis
        feature_names: Optional[List] = None,
        n_epoch: int = 10000,
        n_epoch_inner: int = 2,
        n_folds: int = 5,
        task_type: str = "classification",
        samples: int = 2000,
        prefit: bool = False,
        random_state: int = 0,
    ) -> None:
        enable_reproducible_results(random_state)
        if task_type not in [
            "classification",
            "risk_estimation",
        ]:
            raise RuntimeError(f"Invalid task type {task_type}")

        self.task_type = task_type
        self.feature_names = (
            feature_names if feature_names is not None else pd.DataFrame(X).columns
        )
        self.n_epoch = n_epoch

        super().__init__(self.feature_names)

        model = copy.deepcopy(estimator)

        self.explainer: Union[invaseCV, invaseClassifier, invaseRiskEstimation]
        if task_type in ["classification"]:
            if not prefit:
                model.fit(X, y)
            if n_folds == 1:
                self.explainer = invaseClassifier(
                    model,
                    X,
                    n_epoch=n_epoch,
                    n_epoch_inner=n_epoch_inner,
                    feature_names=self.feature_names,
                )
            else:
                self.explainer = invaseCV(
                    model,
                    X,
                    n_epoch=n_epoch,
                    n_folds=n_folds,
                    n_epoch_inner=n_epoch_inner,
                    feature_names=self.feature_names,
                )
        elif task_type in ["risk_estimation"]:
            if eval_times is None:
                raise RuntimeError("Invalid input for risk estimation interpretability")

            if not prefit:
                if time_to_event is None:
                    raise RuntimeError(
                        "Invalid time_to_event for risk estimation interpretability"
                    )
                model.fit(X, time_to_event, y)

            self.explainer = invaseRiskEstimation(
                model,
                X,
                eval_times=eval_times,
                n_epoch=n_epoch,
                n_epoch_inner=n_epoch_inner,
                samples=samples,
                feature_names=self.feature_names,
            )

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        result = self.explainer.explain(X)

        return np.asarray(result)

    def plot(self, values: pd.DataFrame) -> None:  # type: ignore
        values = pd.DataFrame(values, columns=self.feature_names)
        plt.figure(figsize=(20, 6))
        sns.heatmap(values).set_title("invase")

    @staticmethod
    def name() -> str:
        return "invase"

    @staticmethod
    def pretty_name() -> str:
        return "INVASE"


plugin = INVASEPlugin
