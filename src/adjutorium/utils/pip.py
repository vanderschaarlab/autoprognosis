# stdlib
import subprocess
import sys

# adjutorium absolute
import adjutorium.logger as log


def install(packages: list) -> None:
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except BaseException as e:
            log.error(f"failed to install package {package}: {e}")
