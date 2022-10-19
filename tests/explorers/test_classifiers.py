# third party
from explorers_mocks import MockHook
import pytest
from sklearn.datasets import load_breast_cancer

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.classifiers import ClassifierSeeker
from autoprognosis.utils.metrics import evaluate_auc


def test_sanity() -> None:
    model = ClassifierSeeker(
        study_name="test_classifiers",
        CV=10,
        num_iter=123,
        top_k=5,
        timeout=6,
        metric="aucprc",
        feature_scaling=["scaler"],
        classifiers=["perceptron"],
    )

    assert model.CV == 10
    assert model.num_iter == 123
    assert model.top_k == 5
    assert model.timeout == 6

    assert model.estimators[0].feature_scaling[0].name() == "scaler"
    assert model.estimators[0].name() == "perceptron"


def test_fails() -> None:
    with pytest.raises(ValueError):
        ClassifierSeeker(study_name="test_classifiers", classifiers=["invalid"])

    with pytest.raises(ValueError):
        ClassifierSeeker(study_name="test_classifiers", feature_scaling=["invalid"])

    with pytest.raises(ValueError):
        ClassifierSeeker(study_name="test_classifiers", CV=-1)

    with pytest.raises(ValueError):
        ClassifierSeeker(study_name="test_classifiers", num_iter=-2)

    with pytest.raises(ValueError):
        ClassifierSeeker(study_name="test_classifiers", metric="invalid")


def test_search() -> None:
    X, Y = load_breast_cancer(return_X_y=True, as_frame=True)

    seeker = ClassifierSeeker(
        study_name="test_classifiers",
        num_iter=10,
        top_k=3,
        feature_scaling=["scaler", "minmax_scaler"],
        classifiers=[
            "logistic_regression",
            "lda",
            "qda",
            "perceptron",
        ],
    )
    best_models = seeker.search(X, Y)

    assert len(best_models) == 3

    for model in best_models:
        model.fit(X, Y)

        y_pred = model.predict(X)
        y_pred_orig = model.predict(X)

        assert len(y_pred) == len(y_pred_orig)
        assert len(y_pred) == len(Y)

        y_pred_proba = model.predict_proba(X)

        assert len(y_pred_proba) == len(Y)

        assert evaluate_auc(Y, y_pred_proba)[0] > 0.9


def test_hooks() -> None:
    hook = MockHook()

    X, Y = load_breast_cancer(return_X_y=True, as_frame=True)

    seeker = ClassifierSeeker(
        study_name="test_classifiers",
        num_iter=10,
        top_k=3,
        hooks=hook,
    )

    with pytest.raises(StudyCancelled):
        seeker.search(X, Y)
