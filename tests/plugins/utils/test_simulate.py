# stdlib
from typing import Tuple

# third party
import numpy as np
import pytest

# adjutorium absolute
from adjutorium.plugins.utils.simulate import simulate_nan


def dataset(
    mechanism: str, p_miss: float, n: int = 1000, opt: str = "logistic"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(0)

    p = 4

    mean = np.repeat(0, p)
    cov = 0.5 * (np.ones((p, p)) + np.eye(p))

    x = np.random.multivariate_normal(mean, cov, size=n)
    x_simulated = simulate_nan(x, p_miss, mechanism, opt=opt)

    mask = x_simulated["mask"]
    x_miss = x_simulated["X_incomp"]

    return x, x_miss, mask


@pytest.mark.parametrize("mechanism", ["MAR", "MNAR", "MCAR"])
@pytest.mark.parametrize("p_miss", [0.1, 0.3, 0.5])
def test_simulate_nan(mechanism: str, p_miss: float) -> None:
    orig, miss, mask = dataset(mechanism, p_miss)

    np.testing.assert_array_equal((orig != miss), mask)
    np.testing.assert_array_equal(np.isnan(miss), mask)


@pytest.mark.parametrize("opt", ["logistic", "quantile", "selfmasked"])
def test_simulate_simulate_mnar(opt: str) -> None:
    orig, miss, mask = dataset("MNAR", 0.5, opt=opt)

    np.testing.assert_array_equal((orig != miss), mask)
    np.testing.assert_array_equal(np.isnan(miss), mask)
