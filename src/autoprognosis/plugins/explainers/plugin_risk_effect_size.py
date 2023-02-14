# stdlib
import copy
from typing import Any, List, Optional

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# autoprognosis absolute
from autoprognosis.plugins.explainers.base import ExplainerPlugin
from autoprognosis.utils.distributions import enable_reproducible_results


class RiskEffectSizePlugin(ExplainerPlugin):
    """
    Interpretability plugin based on Risk Effect size and Cohen's D.

    Args:
        estimator: model. The model to explain.
        X: dataframe. Training set
        y: dataframe. Training labels
        task_type: str. classification or risk_estimation
        prefit: bool. If true, the estimator won't be trained.
        n_epoch: int. training epochs
        time_to_event: dataframe. Used for risk estimation tasks.
        eval_times: list. Used for risk estimation tasks.

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
        >>>     "risk_effect_size",
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
        task_type: str = "classification",
        feature_names: Optional[List] = None,
        subsample: int = 10,
        prefit: bool = False,
        effect_size: float = 0.5,
        # risk estimation
        time_to_event: Optional[pd.DataFrame] = None,  # for survival analysis
        eval_times: Optional[List] = None,  # for survival analysis
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        enable_reproducible_results(random_state)
        if task_type not in ["classification", "risk_estimation"]:
            raise RuntimeError("invalid task type")

        self.feature_names = (
            feature_names if feature_names is not None else pd.DataFrame(X).columns
        )

        X = pd.DataFrame(X, columns=self.feature_names)
        model = copy.deepcopy(estimator)
        self.task_type = task_type
        self.effect_size = effect_size

        if task_type == "classification":
            if not prefit:
                model.fit(X, y)

            def model_fn(X: pd.DataFrame) -> pd.DataFrame:
                risk_prob = model.predict_proba(X).values[:, 1]
                return pd.DataFrame(risk_prob)

            self.predict_cbk = model_fn
        elif task_type == "risk_estimation":
            if time_to_event is None or eval_times is None:
                raise RuntimeError("Invalid input for risk estimation interpretability")

            if not prefit:
                model.fit(X, time_to_event, y)

            def model_fn(X: pd.DataFrame) -> pd.DataFrame:
                if eval_times is None:
                    raise RuntimeError(
                        "Invalid input for risk estimation interpretability"
                    )

                res = pd.DataFrame(
                    model.predict(X, eval_times).values, columns=eval_times
                )

                return pd.DataFrame(res[eval_times[-1]])

            self.predict_cbk = model_fn

    # function to calculate Cohen's d for independent samples
    def _cohend(self, d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

        # calculate the means of the samples
        u1, u2 = np.mean(d1), np.mean(d2)
        # calculate the effect size
        return np.abs((u1 - u2) / s)

    def _get_population_shifts(
        self,
        predict_cbk: Any,
        X: pd.DataFrame,
        effect_size: Optional[float] = None,
    ) -> pd.DataFrame:
        if not effect_size:
            effect_size = self.effect_size

        training_preds = predict_cbk(X)
        bins = 2
        buckets = pd.cut(
            training_preds.values.squeeze(),
            bins=bins,
            duplicates="drop",
            labels=range(bins),
        )
        X = X.reset_index(drop=True)

        output = pd.DataFrame([], columns=X.columns)
        index = []
        for bucket in range(bins):
            curr_bucket = X[buckets == bucket]
            other_buckets = X[buckets > bucket]

            if len(curr_bucket) < 2 or len(other_buckets) < 2:
                continue

            diffs = self._cohend(curr_bucket, other_buckets).to_dict()

            heatmaps = pd.DataFrame([[0] * len(X.columns)], columns=X.columns)

            for key in diffs:
                if diffs[key] < effect_size:
                    continue

                heatmaps[key] = diffs[key]

            output = output.append(heatmaps)
            index.append(f"Risk lvl {bucket}")

        output.index = index
        output = output.astype(float)
        output = output.clip(upper=3)

        return output

    def plot(self, X: pd.DataFrame, ax: Any = None) -> None:
        output = self._get_population_shifts(self.predict_cbk, X)
        thresh_line = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

        ignore_empty = []
        for col in output.columns:
            if output[col].sum() == 0:
                ignore_empty.append(col)
        output = output.drop(columns=ignore_empty)
        if len(output.columns) == 0:
            return

        draw_lines = []
        thresh_iter = 0

        for idx, col in enumerate(output.columns):
            if (
                thresh_iter < len(thresh_line)
                and output[col].max() < thresh_line[thresh_iter]
            ):
                thresh_iter += 1
                draw_lines.append(idx)
        draw_lines.append(idx + 1)

        cols = output.columns
        cols = sorted(cols, key=lambda key: output[key].max(), reverse=True)

        output = output[cols]

        renamed_cols = {}
        for idx, col in enumerate(output.columns):
            renamed_cols[col] = f"{col} {idx}"

        output = output.rename(columns=renamed_cols)
        output = output.transpose()

        if ax is None:
            plt.figure(figsize=(4, int(len(output) * 0.5)))

        plot_ax = sns.heatmap(
            output,
            cmap="Reds",
            linewidths=0.5,
            linecolor="black",
            annot=True,
            ax=ax,
        )
        plot_ax.xaxis.set_ticks_position("top")

    def explain(
        self, X: pd.DataFrame, effect_size: Optional[float] = None
    ) -> np.ndarray:
        if not effect_size:
            effect_size = self.effect_size
        X = pd.DataFrame(X, columns=self.feature_names)

        shifts = self._get_population_shifts(self.predict_cbk, X, effect_size)

        max_impact = shifts.max()

        return max_impact[max_impact > effect_size].sort_values(ascending=False)

    @staticmethod
    def name() -> str:
        return "risk_effect_size"

    @staticmethod
    def pretty_name() -> str:
        return "Risk Effect size"


plugin = RiskEffectSizePlugin
