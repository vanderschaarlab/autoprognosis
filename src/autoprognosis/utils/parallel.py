# stdlib
import multiprocessing
import os

# autoprognosis absolute
import autoprognosis.logger as log


def n_opt_jobs() -> int:
    try:
        n_jobs = int(os.environ["N_OPT_JOBS"])
    except BaseException as e:
        log.debug(f"failed to get N_JOBS {e}")
        n_jobs = 2
    log.debug(f"Using {n_jobs} cores for HP")
    return n_jobs


def n_learner_jobs() -> int:
    try:
        n_jobs = int(os.environ["N_LEARNER_JOBS"])
    except BaseException as e:
        n_jobs = multiprocessing.cpu_count()
        log.debug(f"failed to get N_LEARNER_JOBS {e}")
    log.debug(f"Using {n_jobs} cores for learners")
    return n_jobs
