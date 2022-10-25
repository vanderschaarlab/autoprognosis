# stdlib
from typing import Any, Callable, Tuple

# third party
from pydantic import validate_arguments

# autoprognosis absolute
from autoprognosis.explorers.core.optimizers.bayesian import BayesianOptimizer


class Optimizer:
    def __init__(
        self,
        study_name: str,
        estimator: Any,
        evaluation_cbk: Callable,
        optimizer_type: str = "bayesian",
        n_trials: int = 50,
        timeout: int = 60,
    ):
        if optimizer_type not in ["bayesian"]:
            raise RuntimeError(f"Invalid optimizer type {optimizer_type}")

        if optimizer_type == "bayesian":
            self.optimizer = BayesianOptimizer(
                study_name=study_name,
                estimator=estimator,
                evaluation_cbk=evaluation_cbk,
                n_trials=n_trials,
                timeout=timeout,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
    ) -> Tuple[float, dict]:
        return self.optimizer.evaluate()


class EnsembleOptimizer:
    def __init__(
        self,
        study_name: str,
        ensemble_len: int,
        evaluation_cbk: Callable,
        optimizer_type: str = "bayesian",
        n_trials: int = 50,
        timeout: int = 60,
        skip_recap: bool = False,
    ):
        if optimizer_type not in ["bayesian"]:
            raise RuntimeError(f"Invalid optimizer type {optimizer_type}")

        if optimizer_type == "bayesian":
            self.optimizer = BayesianOptimizer(
                study_name=study_name,
                ensemble_len=ensemble_len,
                evaluation_cbk=evaluation_cbk,
                n_trials=n_trials,
                timeout=timeout,
                skip_recap=skip_recap,
            )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
    ) -> Tuple[float, dict]:
        return self.optimizer.evaluate_ensemble()
