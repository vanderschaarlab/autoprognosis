# stdlib
import sys
import warnings

# autoprognosis absolute
import autoprognosis.logger as log

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log.add(sink=sys.stderr, level="ERROR")
