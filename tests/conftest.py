# stdlib
import sys
import warnings

# adjutorium absolute
import adjutorium.logger as log

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log.add(sink=sys.stderr, level="ERROR")
