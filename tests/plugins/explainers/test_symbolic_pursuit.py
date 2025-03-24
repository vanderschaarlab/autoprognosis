# stdlib
from typing import Tuple, Type

import numpy as np
import pandas as pd
import pytest

# third party
from lifelines.datasets import load_rossi
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.plugins.explainers.plugin_symbolic_pursuit import plugin
from autoprognosis.plugins.pipeline import Pipeline
from autoprognosis.plugins.prediction.classifiers import Classifiers
from autoprognosis.plugins.prediction.risk_estimation.plugin_cox_ph import (
    plugin as CoxPH,
)
from autoprognosis.plugins.preprocessors import Preprocessors


def dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = load_breast_cancer(return_X_y=True)

    X = pd.DataFrame(X)
    y = pd.Series(y)

    return train_test_split(X, y, test_size=0.2)


def test_sanity() -> None:
    classifier = "logistic_regression"

    X_train, X_test, y_train, y_test = dataset()

    template = Pipeline(
        [
            Preprocessors().get_type("minmax_scaler").fqdn(),
            Classifiers().get_type(classifier).fqdn(),
        ]
    )
    pipeline = template()

    explainer = plugin(
        pipeline,
        X_train,
        y_train,
        task_type="classification",
        loss_tol=100,
        ratio_tol=2,
        maxiter=10,
        eps=1e-3,
        random_state=1,
    )

    assert explainer.loss_tol == 100
    assert explainer.task_type == "classification"
    assert explainer.ratio_tol == 2
    assert explainer.maxiter == 10
    assert explainer.eps == 1e-3
    assert explainer.random_state == 1


@pytest.mark.slow
@pytest.mark.parametrize("exp", [plugin])
def test_plugin_symbolic_pursuit_classifier_prediction(exp: Type) -> None:
    classifier = "logistic_regression"

    X_train, X_test, y_train, y_test = dataset()

    template = Pipeline(
        [
            Classifiers().get_type(classifier).fqdn(),
        ]
    )
    pipeline = template()

    pipeline.fit(X_train, y_train)

    explainer = exp(pipeline, X_train, y_train, maxiter=5, ratio_tol=1.5, patience=2)

    value_of_inf = explainer.explain(X_test)

    assert len(value_of_inf) == len(X_test)


@pytest.mark.slow
def test_plugin_symbolic_pursuit_survival_prediction() -> None:
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    surv = CoxPH().fit(X, T, Y)

    explainer = plugin(
        surv,
        X,
        Y,
        time_to_event=T,
        eval_times=[
            int(T[Y.iloc[:] == 1].quantile(0.50)),
            int(T[Y.iloc[:] == 1].quantile(0.75)),
        ],
        task_type="risk_estimation",
        maxiter=5,
        ratio_tol=1.5,
        patience=2,
    )

    value_of_inf = explainer.explain(X)

    assert len(value_of_inf) == len(X)
