# stdlib
import multiprocessing
import os

# adjutorium absolute
import adjutorium.logger as log


def cpu_count() -> int:
    try:
        n_jobs = int(os.environ["N_JOBS"])
    except BaseException as e:
        log.info(f"failed to get N_JOBS {e}")
        n_jobs = multiprocessing.cpu_count()
    log.info(f"Using {n_jobs} cores")
    return n_jobs
