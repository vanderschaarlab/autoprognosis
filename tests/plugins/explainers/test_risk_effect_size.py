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
from adjutorium.plugins.explainers.plugin_risk_effect_size import plugin
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
        effect_size=0.1,
        task_type="classification",
    )

    assert explainer.effect_size == 0.1


@pytest.mark.parametrize("exp", [plugin, Explainers().get_type("risk_effect_size")])
def test_plugin_risk_effect_size_classifier_prediction(exp: Type) -> None:
    classifier = "logistic_regression"

    X_train, X_test, y_train, y_test = dataset()

    template = Pipeline(
        [
            Classifiers().get_type(classifier).fqdn(),
        ]
    )
    pipeline = template()

    pipeline.fit(X_train, y_train)

    explainer = exp(pipeline, X_train, y_train)

    value_of_inf = explainer.explain(X_test)

    assert value_of_inf.isna().sum().sum() == 0
    assert (value_of_inf < 0).sum().sum() == 0


@pytest.mark.slow
def test_plugin_risk_effect_size_survival_prediction() -> None:
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
    )

    value_of_inf = explainer.explain(X)

    assert value_of_inf.isna().sum().sum() == 0
    assert (value_of_inf < 0).sum().sum() == 0
