# stdlib
from typing import Tuple

# third party
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

# autoprognosis absolute
from autoprognosis.plugins import Predictions
from autoprognosis.utils.serialization import load_model, save_model


def dataset() -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(1)

    N = 1000
    X = rng.randint(N, size=(N, 3))
    y = rng.randint(2, size=(N))

    return X, y


@pytest.mark.parametrize("plugin", Predictions().list())
def test_serialization(plugin: str) -> None:
    X, y = dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    estimator = Predictions().get(plugin)

    estimator.fit(X_train, y_train)
    estimator.predict(X_test)

    buff = estimator.save()
    estimator_new = Predictions().get(plugin).load(buff)

    estimator_new.predict(X_test)


@pytest.mark.parametrize("plugin", Predictions().list())
def test_pickle(plugin: str) -> None:
    X, y = dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    estimator = Predictions().get(plugin)

    estimator.fit(X_train, y_train)
    estimator.predict(X_test)

    buff = save_model(estimator)
    estimator_new = load_model(buff)

    estimator_new.predict(X_test)
