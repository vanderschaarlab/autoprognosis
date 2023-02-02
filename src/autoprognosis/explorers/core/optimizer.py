# stdlib
from typing import Any, Callable, List, Tuple

# third party
from pydantic import validate_arguments

# autoprognosis absolute
from autoprognosis.explorers.core.optimizers.bayesian import BayesianOptimizer
from autoprognosis.explorers.core.optimizers.hyperband import HyperbandOptimizer


class Optimizer:
    def __init__(
        self,
        study_name: str,
        estimator: Any,
        evaluation_cbk: Callable,
        optimizer_type: str = "bayesian",
        n_trials: int = 50,  # bayesian: number of trials
        timeout: int = 60,  # bayesian: timeout per search
        eta: int = 3,  # hyperband: defines configuration downsampling rate (default = 3)
        random_state: int = 0,
    ):
        if optimizer_type not in ["bayesian", "hyperband"]:
            raise RuntimeError(f"Invalid optimizer type {optimizer_type}")

        if optimizer_type == "bayesian":
            self.optimizer = BayesianOptimizer(
                study_name=study_name,
                estimator=estimator,
                evaluation_cbk=evaluation_cbk,
                n_trials=n_trials,
                timeout=timeout,
                random_state=random_state,
            )
        elif optimizer_type == "hyperband":
            self.optimizer = HyperbandOptimizer(
                study_name=study_name,
                estimator=estimator,
                evaluation_cbk=evaluation_cbk,
                max_iter=n_trials,
                eta=eta,
                random_state=random_state,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
    ) -> Tuple[List[float], List[dict]]:
        return self.optimizer.evaluate()


class EnsembleOptimizer:
    def __init__(
        self,
        study_name: str,
        ensemble_len: int,
        evaluation_cbk: Callable,
        optimizer_type: str = "bayesian",
        n_trials: int = 50,  # bayesian: number of trials
        timeout: int = 60,  # bayesian: timeout per search
        max_iter: int = 27,  # hyperband: maximum iterations per configuration
        eta: int = 3,  # hyperband: defines configuration downsampling rate (default = 3)
        skip_recap: bool = False,
        random_state: int = 0,
    ):
        if optimizer_type not in ["bayesian", "hyperband"]:
            raise RuntimeError(f"Invalid optimizer type {optimizer_type}")

        if optimizer_type == "bayesian":
            self.optimizer = BayesianOptimizer(
                study_name=study_name,
                ensemble_len=ensemble_len,
                evaluation_cbk=evaluation_cbk,
                n_trials=n_trials,
                timeout=timeout,
                skip_recap=skip_recap,
                random_state=random_state,
            )
        elif optimizer_type == "hyperband":
            self.optimizer = HyperbandOptimizer(
                study_name=study_name,
                ensemble_len=ensemble_len,
                evaluation_cbk=evaluation_cbk,
                max_iter=max_iter,
                eta=eta,
                random_state=random_state,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
    ) -> Tuple[float, dict]:
        return self.optimizer.evaluate_ensemble()
