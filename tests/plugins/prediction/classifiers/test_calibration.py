# stdlib
from typing import Tuple

# third party
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.plugins.prediction import Predictions


def gen_dataset() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    n_samples = 5000
    centers = [(-5, -5), (0, 0), (5, 5)]
    X, y = make_blobs(
        n_samples=n_samples, centers=centers, shuffle=False, random_state=42
    )

    y[: n_samples // 2] = 0
    y[n_samples // 2 :] = 1
    sample_weight = np.random.RandomState(42).rand(y.shape[0])

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weight, test_size=0.9, random_state=42
    )

    return X_train, X_test, y_train, y_test, sw_train, sw_test


def supports_calibration(plugin: str) -> bool:
    test_plugin = Predictions().get(plugin)

    if len(test_plugin.hyperparameter_space()) == 0:
        return False

    for hp in test_plugin.hyperparameter_space():
        if hp.name == "calibration":
            return True

    return False


@pytest.mark.parametrize("plugin", Predictions().list())
def test_plugin_calibration(plugin: str) -> None:
    if not supports_calibration(plugin):
        return

    X_train, X_test, y_train, y_test, sw_train, sw_test = gen_dataset()

    test_plugin = Predictions().get(plugin, calibration=0)
    test_plugin.fit(X_train, y_train)

    prob_no_cal = test_plugin.predict_proba(X_test).to_numpy()[:, 1]

    score_no_cal = brier_score_loss(y_test, prob_no_cal, sample_weight=sw_test)

    for method in [0, 1, 2]:
        test_plugin = Predictions().get(plugin, calibration=method)
        test_plugin.fit(X_train, y_train)

        probs = test_plugin.predict_proba(X_test).to_numpy()[:, 1]
        score = brier_score_loss(y_test, probs, sample_weight=sw_test)

        print(
            f"score without calibration {score_no_cal} score with calibration {score}"
        )
