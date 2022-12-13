# stdlib
import multiprocessing
import os

# autoprognosis absolute
from autoprognosis.utils.parallel import n_learner_jobs, n_opt_jobs


def test_n_opt_jobs() -> None:
    os.environ["N_OPT_JOBS"] = "1"

    assert n_opt_jobs() == 1

    del os.environ["N_OPT_JOBS"]

    assert n_opt_jobs() == multiprocessing.cpu_count()


def test_n_learner_jobs() -> None:
    os.environ["N_LEARNER_JOBS"] = "1"

    assert n_learner_jobs() == 1

    del os.environ["N_LEARNER_JOBS"]

    assert n_learner_jobs() == 2
