# stdlib
import os
import sys
import warnings

# third party
import optuna

# autoprognosis relative
from . import logger  # noqa: F401

optuna.logging.set_verbosity(optuna.logging.FATAL)
optuna.logging.disable_propagation()
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


logger.add(sink=sys.stderr, level="CRITICAL")

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
