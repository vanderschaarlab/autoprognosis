# stdlib
import os
import sys
import warnings

# adjutorium relative
from . import logger  # noqa: F401

logger.add(sink=sys.stderr, level="CRITICAL")

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
