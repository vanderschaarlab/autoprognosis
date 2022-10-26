# third party
from explorers_mocks import MockHook
import pytest
from sklearn.datasets import load_diabetes

# autoprognosis absolute
from autoprognosis.exceptions import StudyCancelled
from autoprognosis.explorers.regression import RegressionSeeker


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_sanity(optimizer_type: str) -> None:
    model = RegressionSeeker(
        study_name="test_regressors",
        CV=10,
        num_iter=123,
        top_k=5,
        timeout=6,
        metric="r2",
        feature_scaling=["scaler"],
        regressors=["linear_regression"],
        optimizer_type=optimizer_type,
    )

    assert model.CV == 10
    assert model.num_iter == 123
    assert model.top_k == 5
    assert model.timeout == 6

    assert model.estimators[0].feature_scaling[0].name() == "scaler"
    assert model.estimators[0].name() == "linear_regression"


def test_fails() -> None:
    with pytest.raises(ValueError):
        RegressionSeeker(study_name="test_regressors", regressors=["invalid"])

    with pytest.raises(ValueError):
        RegressionSeeker(study_name="test_regressors", feature_scaling=["invalid"])

    with pytest.raises(ValueError):
        RegressionSeeker(study_name="test_regressors", CV=-1)

    with pytest.raises(ValueError):
        RegressionSeeker(study_name="test_regressors", num_iter=-2)

    with pytest.raises(ValueError):
        RegressionSeeker(study_name="test_regressors", metric="invalid")


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_search(optimizer_type: str) -> None:
    X, Y = load_diabetes(return_X_y=True, as_frame=True)

    seeker = RegressionSeeker(
        study_name="test_regressors",
        num_iter=10,
        top_k=3,
        feature_scaling=["scaler", "minmax_scaler"],
        regressors=[
            "linear_regression",
            "random_forest_regressor",
        ],
        optimizer_type=optimizer_type,
    )
    best_models = seeker.search(X, Y)

    assert len(best_models) == 2

    for model in best_models:
        model.fit(X, Y)

        y_pred = model.predict(X)
        y_pred_orig = model.predict(X)

        assert len(y_pred) == len(y_pred_orig)
        assert len(y_pred) == len(Y)


@pytest.mark.parametrize("optimizer_type", ["bayesian", "hyperband"])
def test_hooks(optimizer_type: str) -> None:
    hook = MockHook()

    X, Y = load_diabetes(return_X_y=True, as_frame=True)

    seeker = RegressionSeeker(
        study_name="test_regressors",
        num_iter=10,
        top_k=3,
        hooks=hook,
        optimizer_type=optimizer_type,
    )

    with pytest.raises(StudyCancelled):
        seeker.search(X, Y)
