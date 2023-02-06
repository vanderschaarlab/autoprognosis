# stdlib
import copy
from typing import Any, Callable, List, Optional, Tuple

# third party
import numpy as np
import optuna
from pydantic import validate_arguments

# autoprognosis absolute
import autoprognosis.logger as log
from autoprognosis.utils.redis import RedisBackend

threshold = 100
EPS = 1e-8


class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    pass


class ParamRepeatPruner:
    """Prunes reapeated trials, which means trials with the same paramters won't waste time/resources."""

    def __init__(
        self,
        study: optuna.study.Study,
        patience: int,
    ) -> None:
        self.study = study
        self.seen: set = set()

        self.best_score: float = -1
        self.no_improvement_for = 0
        self.patience = patience

        if self.study is not None:
            self.register_existing_trials()

    def register_existing_trials(self) -> None:
        for trial_idx, trial_past in enumerate(
            self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        ):
            if trial_past.values[0] > self.best_score:
                self.best_score = trial_past.values[0]
                self.no_improvement_for = 0
            else:
                self.no_improvement_for += 1
            self.seen.add(hash(frozenset(trial_past.params.items())))

    def check_patience(
        self,
        trial: optuna.trial.Trial,
    ) -> None:
        if self.no_improvement_for > self.patience:
            raise EarlyStoppingExceeded()

    def check_trial(
        self,
        trial: optuna.trial.Trial,
    ) -> None:
        self.check_patience(trial)

        params = frozenset(trial.params.items())

        current_val = hash(params)
        if current_val in self.seen:
            raise optuna.exceptions.TrialPruned()

        self.seen.add(current_val)

    def report_score(self, score: float) -> None:
        if score > self.best_score:
            self.best_score = score
            self.no_improvement_for = 0
        else:
            self.no_improvement_for += 1


class BayesianOptimizer:
    """Optimization helper based on Bayesian Optimization.

    Args:
        patience: int
            maximum iterations without any gain
        random_state: int
            random seed
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        study_name: str,
        evaluation_cbk: Callable,
        estimator: Any = None,
        ensemble_len: Optional[int] = None,
        n_trials: int = 50,
        timeout: int = 60,
        skip_recap: bool = False,
        random_state: int = 0,
    ):
        self.study_name = study_name
        self.estimator = estimator
        self.ensemble_len = ensemble_len
        self.evaluation_cbk = evaluation_cbk
        self.n_trials = n_trials
        self.timeout = timeout
        self.skip_recap = skip_recap
        self.random_state = random_state

    def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
        load_if_exists: bool = True,
        storage_type: str = "redis",
        patience: int = threshold,
    ) -> Tuple[optuna.Study, ParamRepeatPruner]:
        """Helper for creating a new study.

        Args:
            study_name: str
                Study ID
            direction: str
                maximize/minimize
            load_if_exists: bool
                If True, it tries to load previous trials from the storage.
            storage_type: str
                redis/none
            patience: int
                How many trials without improvement to accept.

        """

        storage_obj = None
        if storage_type == "redis":
            try:
                backend = RedisBackend()
                storage_obj = backend.optuna()
            except BaseException:
                storage_obj = None

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        try:
            study = optuna.create_study(
                direction=direction,
                study_name=study_name,
                storage=storage_obj,
                load_if_exists=load_if_exists,
                sampler=sampler,
            )
        except BaseException as e:
            log.debug(f"create_study failed {e}")
            study = optuna.create_study(
                direction=direction,
                study_name=study_name,
                sampler=sampler,
            )

        return study, ParamRepeatPruner(study, patience=patience)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(
        self,
    ) -> Tuple[List[float], List[dict]]:
        if self.estimator is None:
            raise ValueError("Invalid estimator")
        study, pruner = self.create_study(study_name=self.study_name)

        baseline_score = self.evaluation_cbk()
        pruner.report_score(baseline_score)

        log.info(f"baseline score for {self.estimator.name()} {baseline_score}")

        if len(self.estimator.hyperparameter_space()) == 0:
            return [baseline_score], [{}]

        def objective(trial: optuna.Trial) -> float:
            args = self.estimator.sample_hyperparameters(trial)
            pruner.check_trial(trial)

            score = self.evaluation_cbk(**args)

            pruner.report_score(score)

            return score

        try:
            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        except EarlyStoppingExceeded:
            log.info("Early stopping triggered for search")

        scores = [baseline_score]
        params = [{}]
        for trial_idx, trial_past in enumerate(
            study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        ):
            scores.append(trial_past.values[0])
            params.append(trial_past.params)

        return scores, params

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_ensemble(
        self,
    ) -> Tuple[float, dict]:
        if self.ensemble_len is None:
            raise ValueError("Invalid ensemble len")

        study, pruner = self.create_study(
            study_name=self.study_name, load_if_exists=False, storage_type="none"
        )

        def objective(trial: optuna.Trial) -> float:
            weights = [
                trial.suggest_int(f"weight_{idx}", 0, 10)
                for idx in range(self.ensemble_len)
            ]
            pruner.check_trial(trial)
            weights = weights / (np.sum(weights) + EPS)

            score = self.evaluation_cbk(weights)

            pruner.report_score(score)

            return score

        if not self.skip_recap:
            initial_trials = []

            trial_template = {}
            for idx in range(self.ensemble_len):
                trial_template[f"weight_{idx}"] = 0

            for idx in range(self.ensemble_len):
                local_trial = copy.deepcopy(trial_template)
                local_trial[f"weight_{idx}"] = 1
                initial_trials.append(local_trial)

            for trial in initial_trials:
                study.enqueue_trial(trial)

        try:
            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        except EarlyStoppingExceeded:
            log.info("Early stopping triggered for search")

        return study.best_value, study.best_trial.params
