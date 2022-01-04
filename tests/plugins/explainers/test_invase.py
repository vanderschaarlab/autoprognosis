# stdlib
from typing import Tuple, Type

# third party
from lifelines.datasets import load_rossi
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# adjutorium absolute
from adjutorium.plugins.explainers import Explainers
from adjutorium.plugins.explainers.data_generation import generate_dataset
from adjutorium.plugins.explainers.plugin_invase import plugin
from adjutorium.plugins.pipeline import Pipeline
from adjutorium.plugins.prediction.classifiers import Classifiers
from adjutorium.plugins.prediction.risk_estimation.plugin_cox_ph import plugin as CoxPH
from adjutorium.plugins.preprocessors import Preprocessors


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
        n_epoch=2,
        n_epoch_inner=0,
        task_type="classification",
    )

    assert explainer.n_epoch == 2


@pytest.mark.parametrize("exp", [plugin, Explainers().get_type("invase")])
def test_plugin_invase_classifier_prediction(exp: Type) -> None:
    classifier = "logistic_regression"

    X_train, X_test, y_train, y_test = dataset()

    template = Pipeline(
        [
            Classifiers().get_type(classifier).fqdn(),
        ]
    )
    pipeline = template()

    pipeline.fit(X_train, y_train)

    explainer = exp(pipeline, X_train, y_train, n_epoch=150, n_folds=1, n_epoch_inner=1)

    selection_prob = np.asarray(explainer.explain(X_test))

    assert selection_prob.max() <= 1
    assert selection_prob.min() >= 0


@pytest.mark.slow
def test_plugin_invase_survival_prediction() -> None:
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
        n_epoch=250,
        n_epoch_inner=1,
        n_folds=1,
        task_type="risk_estimation",
    )

    result = explainer.explain(X)

    print("result ", result[0, :])
    assert result.shape == (X.shape[0], X.shape[1], 2)


@pytest.mark.parametrize("data_type", ["syn2", "syn3", "syn6"])
@pytest.mark.slow
def test_synthetic_dataset(data_type: str) -> None:
    classifier = "logistic_regression"

    template = Pipeline(
        [
            Classifiers().get_type(classifier).fqdn(),
        ]
    )
    baseline = template()

    x_train, y_train, g_train = generate_dataset(
        n=10000, dim=11, data_type=data_type, seed=0
    )
    x_test, y_test, g_test = generate_dataset(
        n=1000, dim=11, data_type=data_type, seed=0
    )
    y_train = y_train[:, 0].astype(int)
    y_test = y_test[:, 0].astype(int)

    explainer = plugin(
        baseline, x_train, y_train, n_epoch=150, n_folds=1, n_epoch_inner=1
    )

    g_hat = explainer.explain(x_test)
    g_hat = np.asarray(g_hat)
    assert g_hat.max() <= 1
    assert g_hat.min() >= 0
