# stdlib
from pathlib import Path
import subprocess
import sys

# autoprognosis absolute
import autoprognosis.logger as log

current_dir = Path(__file__).parent

predefined = {}


def install(packages: list) -> None:
    for package in packages:
        install_pack = package
        if package in predefined:
            install_pack = predefined[package]
        log.error(f"Installing {install_pack}")

        try:
            subprocess.check_call(
                ["pip", "install", install_pack],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            continue
        except BaseException as e:
            log.error(f"failed to install package {package} from pip: {e}")

        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", install_pack],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            continue
        except BaseException as e:
            log.error(f"failed to install package {package} from python -m pip: {e}")
