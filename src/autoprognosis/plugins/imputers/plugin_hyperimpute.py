# stdlib
from typing import Any, List

# third party
from hyperimpute.plugins.imputers.plugin_hyperimpute import plugin as base_model

# autoprognosis absolute
import autoprognosis.plugins.core.params as params
import autoprognosis.plugins.imputers.base as base


class HyperImputePlugin(base.ImputerPlugin):
    """ "HyperImpute strategy, a generalized iterative imputation framework for adaptively and automatically configuring column-wise models and their hyperparameters.

    Args:
        classifier_seed: list.
            List of ClassifierPlugin names for the search pool.
        regression_seed: list.
            List of RegressionPlugin names for the search pool.
        imputation_order: int.
            0 - ascending, 1 - descending, 2 - random
        baseline_imputer: int.
            0 - mean, 1 - median, 2- most_frequent
        optimizer: str.
            Hyperparam search strategy. Options: simple, hyperband, bayesian
        class_threshold: int.
            Maximum number of unique items in a categorical column.
        optimize_thresh: int.
            The number of subsamples used for the model search.
        n_inner_iter: int.
            number of imputation iterations.
        select_model_by_column: bool.
            If False, reuse the first model selected in the current iteration for all columns. Else, search the      model for each column.
        select_model_by_iteration: bool.
            If False, reuse the models selected in the first iteration. Otherwise, refresh the models on each        iteration.
        select_lazy: bool.
            If True, if there is a trend towards a certain model architecture, the loop reuses than for all          columns, instead of calling the optimizer.
        inner_loop_hook: Callable.
            Debug hook, called before each iteration.
        random_state: int.
            random seed.

    Example:
        >>> import numpy as np
        >>> from autoprognosis.plugins.imputers import Imputers
        >>> plugin = Imputers().get("hyperimpute")
        >>> plugin.fit_transform([[1, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan], [1, 2, 2, 1], [2, 2, 2, 2]])


    Reference: "HyperImpute: Generalized Iterative Imputation with Automatic Model Selection" """

    def __init__(self, random_state: int = 0, **kwargs: Any) -> None:
        model = base_model(random_state=random_state, **kwargs)

        super().__init__(model)

    @staticmethod
    def name() -> str:
        return base_model.name()

    @staticmethod
    def hyperparameter_space(*args: Any, **kwargs: Any) -> List[params.Params]:
        return base_model.hyperparameter_space()


plugin = HyperImputePlugin
