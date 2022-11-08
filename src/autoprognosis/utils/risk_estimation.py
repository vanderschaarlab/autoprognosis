# stdlib
from typing import Any, Tuple

# third party
from lifelines import CRCSplineFitter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_dataset_for_horizon(
    X: pd.DataFrame, T: pd.DataFrame, Y: pd.DataFrame, horizon_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate the dataset at a certain time horizon. Useful for classifiers.

    Args:
        X: pd.DataFrame, the feature set
        T: pd.DataFrame, days to event or censoring
        Y: pd.DataFrame, outcome or censoring
        horizon_days: int, days to the expected horizon

    Returns:
        X: the feature set for that horizon
        T: days to event or censoring
        Y: Outcome or censoring

    """

    X = X.copy().reset_index(drop=True)
    T = T.copy().reset_index(drop=True)
    Y = Y.copy().reset_index(drop=True)

    event_horizon = ((Y == 1) & (T <= horizon_days)) | ((Y == 0) & (T > horizon_days))
    censored_event_horizon = (Y == 1) & (T > horizon_days)

    X_horizon = X[event_horizon]
    X_horizon_cens = X[censored_event_horizon]

    Y_horizon = Y[event_horizon]
    Y_horizon_cens = 1 - Y[censored_event_horizon]

    T_horizon = T[event_horizon]
    T_horizon_cens = T[censored_event_horizon]

    return (
        pd.concat([X_horizon, X_horizon_cens], ignore_index=True),
        pd.concat([T_horizon, T_horizon_cens], ignore_index=True),
        pd.concat([Y_horizon, Y_horizon_cens], ignore_index=True),
    )


def survival_probability_calibration(
    name: str,
    y_pred: pd.DataFrame,
    T_test: pd.DataFrame,
    y_test: pd.DataFrame,
    t0: float,
    ax: Any,
    color: str,
) -> Tuple:
    """
    Smoothed calibration curves for time-to-event models. This is analogous to
    calibration curves for classification models, extended to handle survival probabilities
    and censoring. Produces a matplotlib figure and some metrics.

    We want to calibrate our model's prediction of :math:`P(T < \text{t0})` against the observed frequencies.

    """

    def ccl(p: np.ndarray) -> np.ndarray:
        return np.log(-np.log(1 - p))

    if ax is None:
        ax = plt.gca()

    predictions_at_t0 = np.clip(y_pred.T.squeeze(), 1e-10, 1 - 1e-10)

    # create new dataset with the predictions
    prediction_df = pd.DataFrame(
        {"ccl_at_%d" % t0: ccl(predictions_at_t0), "time": T_test, "event": y_test}
    )

    # fit new dataset to flexible spline model
    # this new model connects prediction probabilities and actual survival. It should be very flexible, almost to the point of overfitting. It's goal is just to smooth out the data!
    knots = 3
    regressors = {
        "beta_": ["ccl_at_%d" % t0],
        "gamma0_": "1",
        "gamma1_": "1",
        "gamma2_": "1",
    }

    # this model is from examples/royson_crowther_clements_splines.py
    crc = CRCSplineFitter(knots, penalizer=0.000001)
    crc.fit_right_censoring(prediction_df, "time", "event", regressors=regressors)

    # predict new model at values 0 to 1, but remember to ccl it!
    x = np.linspace(
        np.clip(predictions_at_t0.min() - 0.01, 0, 1),
        np.clip(predictions_at_t0.max() + 0.01, 0, 1),
        100,
    )
    y = (
        1
        - crc.predict_survival_function(
            pd.DataFrame({"ccl_at_%d" % t0: ccl(x)}), times=[t0]
        ).T.squeeze()
    )

    # plot our results
    ax.set_title(
        "Smoothed calibration curve of \npredicted vs observed probabilities of t ≤ %d mortality"
        % t0
    )

    ax.plot(x, y, label=name, color=color)
    ax.set_xlabel("Predicted probability of \nt ≤ %d mortality" % t0)
    ax.set_ylabel("Observed probability of \nt ≤ %d mortality" % t0, color=color)
    ax.tick_params(axis="y", labelcolor=color)

    # plot x=y line
    ax.plot(x, x, c="k", ls="--")
    ax.legend()

    # plot histogram of our original predictions
    color = "tab:blue"
    twin_ax = ax.twinx()
    twin_ax.set_ylabel(
        "Count of \npredicted probabilities", color=color
    )  # we already handled the x-label with ax1
    twin_ax.tick_params(axis="y", labelcolor=color)
    twin_ax.hist(predictions_at_t0, alpha=0.3, bins="sqrt", color=color)

    plt.tight_layout()

    deltas = (
        (1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze()
        - predictions_at_t0
    ).abs()
    ICI = deltas.mean()
    E50 = np.percentile(deltas, 50)

    return ax, ICI, E50
