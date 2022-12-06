# stdlib
import random

# third party
import numpy as np


def enable_reproducible_results(seed: int = 0) -> None:
    """Set fixed seed for all the libraries"""
    random.seed(seed)

    np.random.seed(seed)
