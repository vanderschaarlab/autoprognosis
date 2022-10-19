# third party
import numpy as np

# autoprognosis absolute
from autoprognosis.plugins.utils.metrics import MAE, RMSE


def test_MAE() -> None:
    data = np.array([1, 2, 3])
    data_truth = np.array([1, 2, 4])
    mask = np.array([False, True, True])
    assert MAE(data, data_truth, mask) == 0.5


def test_RMSE() -> None:
    data = np.array([1, 2, 3])
    data_truth = np.array([1, 2, 5])
    mask = np.array([False, False, True])
    assert RMSE(data, data_truth, mask) == 2
