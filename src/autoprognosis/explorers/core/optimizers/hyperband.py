# stdlib
import json
import math
from typing import Any, Callable, List, Optional, Set, Tuple

# third party
import numpy as np
from pydantic import validate_arguments

# autoprognosis absolute
import autoprognosis.logger as log

EPS = 1e-8


class HyperbandOptimizer:
    """Optimization helper based on HyperBand.

    Args:
        name: str
            ID
        category: str
            classification/regression. Impacts the objective function
        classifier_seed: list
            List of classification methods to evaluate. Used when category = "classifier"
        regression_seed: list
            List of regression methods to evaluate. Used when category = "regression"
        max_iter: int
            maximum iterations per configuration
        eta: int
            configuration downsampling rate
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
        max_iter: int = 81,  # maximum iterations per configuration
        eta: int = 3,  # defines configuration downsampling rate (default = 3)
        random_state: int = 0,
    ) -> None:
        self.study_name = study_name
        self.random_state = random_state
        self.max_iter = max_iter
        self.eta = eta
        self.estimator = estimator
        self.ensemble_len = ensemble_len
        self.evaluation_cbk = evaluation_cbk

        def logeta(x: Any) -> Any:
            return math.log(x) / math.log(self.eta)

        self.logeta = logeta
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self._reset()

    def _reset(self) -> None:
        self.visited: Set[str] = set()

    def _hash_dict(self, dict_val: dict) -> str:
        return json.dumps(dict_val, sort_keys=True, cls=NpEncoder)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _sample(self, n: int) -> list:
        configurations = []

        for t in range(n):
            if self.estimator is not None:
                params = self.estimator.sample_hyperparameters_np()
            elif self.ensemble_len is not None:
                params = np.random.rand(self.ensemble_len)
                params = params / (np.sum(params) + EPS)
            else:
                raise RuntimeError("need to provide estimator of ensemble len")

            hashed = self._hash_dict(params)

            if hashed in self.visited:
                continue

            self.visited.add(hashed)
            configurations.append(params)

        return configurations

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _internal_evaluate(
        self, objective: Callable, candidate: dict
    ) -> Tuple[float, dict]:
        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(math.ceil(self.B / self.max_iter / (s + 1) * self.eta**s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = self._sample(math.ceil(n))

            for i in range(s + 1):  # changed from s + 1
                if len(T) == 0:
                    break

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations
                n_configs = int(math.ceil(n * self.eta ** (-i)))
                n_iterations = r * self.eta ** (i)

                scores = []

                for model_params in T:
                    score = objective(
                        hyperparam_search_iterations=n_iterations,
                        model_params=model_params,
                    )
                    if score > candidate["score"]:
                        candidate = {
                            "score": score,
                            "params": model_params,
                        }
                    scores.append(score)
                # select a number of best configurations for the next loop
                # filter out early stops, if any
                saved = int(math.ceil(n_configs / self.eta))

                indices = np.argsort(scores)
                T = [T[i] for i in indices]
                T = T[-saved:]
                scores = [scores[i] for i in indices]
                scores = scores[-saved:]

        log.info(
            f"      >>> {self.study_name} -- best candidate: ({candidate['params']}) --- score : {candidate['score']}"
        )

        return candidate["score"], candidate["params"]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate(self) -> Tuple[List[float], List[dict]]:
        baseline_score = self.evaluation_cbk()
        candidate = {
            "score": baseline_score,
            "params": {},
        }

        def objective(hyperparam_search_iterations: int, model_params: dict) -> float:
            return self.evaluation_cbk(
                hyperparam_search_iterations=hyperparam_search_iterations,
                random_state=self.random_state,
                **model_params,
            )

        score, params = self._internal_evaluate(objective, candidate)

        return [score], [params]

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def evaluate_ensemble(self) -> Tuple[float, dict]:
        self._reset()

        candidate = {
            "score": 0,
            "params": {},
        }

        for pos in range(self.ensemble_len):
            weights = np.zeros(self.ensemble_len)
            weights[pos] = 1

            baseline_score = self.evaluation_cbk(weights)

            if baseline_score > candidate["score"]:
                candidate = {
                    "score": baseline_score,
                    "params": weights,
                }

        log.info(f"Baseline ensemble candidate: {candidate}")

        def objective(hyperparam_search_iterations: int, model_params: dict) -> float:
            return self.evaluation_cbk(
                model_params,
            )

        score, weights = self._internal_evaluate(objective, candidate)

        out_weights = {}

        for idx, w in enumerate(weights):
            out_weights[f"weight_{idx}"] = w

        return score, out_weights


class NpEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool):
            return int(obj)
        return super(NpEncoder, self).default(obj)
