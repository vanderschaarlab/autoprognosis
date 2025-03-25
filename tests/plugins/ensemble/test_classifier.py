# stdlib
from typing import Any, List

# third party
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.plugins.ensemble.classifiers import (
    AggregatingEnsemble,
    StackingEnsemble,
    WeightedEnsemble,
    WeightedEnsembleCV,
)
from autoprognosis.plugins.pipeline import Pipeline
from autoprognosis.utils.metrics import (
    evaluate_auc,
    evaluate_brier_score,
    evaluate_c_index,
)


@pytest.mark.parametrize("serialize", [False, True])
def test_weighted_ensemble_sanity(serialize: bool) -> None:
    dtype = Pipeline(
        ["imputer.default.ice", "prediction.classifier.logistic_regression"]
    )
    dtype2 = Pipeline(["prediction.classifier.xgboost"])

    ens = WeightedEnsemble([dtype(), dtype2()], [0.5, 0.5], explainer_plugins=[])

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    ens.fit(X_train, y_train)

    y_pred = ens.predict_proba(X_test)
    assert evaluate_auc(y_test, y_pred.to_numpy())[0] > 0.5

    if serialize:
        buff = ens.save()
        ens = WeightedEnsemble.load(buff)

    y_pred = ens.predict_proba(X_test)

    assert evaluate_auc(y_test, y_pred.to_numpy())[0] > 0.5


@pytest.mark.slow
def test_weighted_ensemble_explainer() -> None:
    dtype = Pipeline(
        ["imputer.default.ice", "prediction.classifier.logistic_regression"]
    )

    ens = WeightedEnsemble(
        [dtype()],
        [0.99],
        explainer_plugins=["invase", "kernel_shap"],
        explanations_nepoch=10,
    )

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    ens.fit(X_train, y_train)

    y_pred = ens.explain(X_test.head(3))
    print("y_pred", y_pred)
    assert sorted(y_pred.keys()) == sorted(["invase", "kernel_shap"])
    for src in y_pred:
        print("weighted_ensemble", src)
        print(y_pred[src].shape)
        assert y_pred[src].shape == (3, X_test.shape[1])


@pytest.mark.parametrize("serialize", [False, True])
def test_weighted_ensemble_cv_sanity(serialize: bool) -> None:
    dtype = Pipeline(
        ["imputer.default.ice", "prediction.classifier.logistic_regression"]
    )
    dtype2 = Pipeline(["prediction.classifier.xgboost"])

    ens = WeightedEnsembleCV(
        models=[dtype(), dtype2()],
        weights=[0.5, 0.5],
        explainer_plugins=[],
        n_folds=2,
    )

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    ens.fit(X_train, y_train)

    y_pred = ens.predict_proba(X_test)
    assert evaluate_auc(y_test, y_pred.to_numpy())[0] > 0.5

    if serialize:
        buff = ens.save()
        ens = WeightedEnsembleCV.load(buff)

    y_pred = ens.predict_proba(X_test)

    assert evaluate_auc(y_test, y_pred.to_numpy())[0] > 0.5

    y_pred, uncert = ens.predict_proba_with_uncertainity(X_test)

    assert uncert.shape == (len(y_pred), 1)
    assert y_pred.shape == (len(X_test), 2)


@pytest.mark.slow
def test_weighted_ensemble_cv_explainer() -> None:
    dtype = Pipeline(
        ["imputer.default.ice", "prediction.classifier.logistic_regression"]
    )

    ens = WeightedEnsembleCV(
        models=[dtype()],
        weights=[0.99],
        explainer_plugins=["invase", "kernel_shap"],
        explanations_nepoch=10,
        n_folds=2,
    )

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    ens.fit(X_train, y_train)

    y_pred = ens.explain(X_test[:3])
    assert sorted(y_pred.keys()) == sorted(["invase", "kernel_shap"])
    for src in y_pred:
        assert y_pred[src].shape == (3, X_test.shape[1])


@pytest.mark.parametrize("serialize", [False, True])
def test_stacked_ensemble_sanity(serialize: bool) -> None:
    dtype = Pipeline(
        ["imputer.default.ice", "prediction.classifier.logistic_regression"]
    )
    dtype2 = Pipeline(["prediction.classifier.xgboost"])
    meta = Pipeline(["prediction.classifier.logistic_regression"])
    ens = StackingEnsemble(
        [dtype(output="numpy"), dtype2(output="numpy")],
        meta(output="numpy"),
        explainer_plugins=[],
    )

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    ens.fit(X_train, y_train)

    y_pred = ens.predict_proba(X_test)

    assert evaluate_auc(y_test, y_pred.to_numpy())[0] > 0.5

    if serialize:
        buff = ens.save()
        ens = StackingEnsemble.load(buff)

    y_pred = ens.predict_proba(X_test)

    assert evaluate_auc(y_test, y_pred.to_numpy())[0] > 0.5


@pytest.mark.slow
def test_stacked_ensemble_explainer() -> None:
    dtype = Pipeline(["prediction.classifier.logistic_regression"])
    meta = Pipeline(["prediction.classifier.logistic_regression"])
    ens = StackingEnsemble(
        [dtype(output="numpy"), dtype(output="numpy")],
        meta(output="numpy"),
        explainer_plugins=["invase", "kernel_shap"],
        explanations_nepoch=10,
    )

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    ens.fit(X_train, y_train)

    y_pred = ens.explain(X_test[:3])
    assert sorted(y_pred.keys()) == sorted(["invase", "kernel_shap"])
    for src in y_pred:
        assert y_pred[src].shape == (3, X_test.shape[1])


@pytest.mark.parametrize("serialize", [False, True])
def test_aggregating_ensemble_sanity(serialize: bool) -> None:
    dtype = Pipeline(
        ["imputer.default.ice", "prediction.classifier.logistic_regression"]
    )
    dtype2 = Pipeline(["prediction.classifier.xgboost"])
    ens = AggregatingEnsemble(
        [dtype(output="pandas"), dtype2(output="pandas")], explainer_plugins=[]
    )

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    ens.fit(X_train, y_train)

    y_pred = ens.predict_proba(X_test)

    assert evaluate_auc(y_test, y_pred.to_numpy())[0] > 0.5

    if serialize:
        buff = ens.save()
        ens = AggregatingEnsemble.load(buff)

    y_pred = ens.predict_proba(X_test)

    assert evaluate_auc(y_test, y_pred.to_numpy())[0] > 0.5


@pytest.mark.slow
def test_aggregating_ensemble_explainer() -> None:
    dtype = Pipeline(["prediction.classifier.logistic_regression"])
    ens = AggregatingEnsemble(
        [dtype(output="pandas"), dtype(output="pandas")],
        explainer_plugins=["invase", "kernel_shap"],
        explanations_nepoch=10,
    )

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    ens.fit(X_train, y_train)

    y_pred = ens.explain(X_test[:3])
    assert sorted(y_pred.keys()) == sorted(["invase", "kernel_shap"])
    for src in y_pred:
        assert y_pred[src].shape == (3, X_test.shape[1])


def helper_eval_survival(
    estimator: Any, eval_time_horizons: List, train_sets: List, test_sets: List
) -> None:
    for e_idx, eval_time in enumerate(eval_time_horizons):
        hX_train, hT_train, hY_train = train_sets[e_idx]
        hX_test, hT_test, hY_test = test_sets[e_idx]

        y_pred = estimator.predict(hX_test, [eval_time]).to_numpy()

        c_index = evaluate_c_index(
            hT_train,
            hY_train,
            y_pred,
            hT_test,
            hY_test,
            eval_time,
        )
        assert c_index > 0.5

        brier = evaluate_brier_score(
            hT_train,
            hY_train,
            y_pred,
            hT_test,
            hY_test,
            eval_time,
        )
        assert brier < 0.5
        print("perf metrics ", estimator.name(), c_index, brier)
