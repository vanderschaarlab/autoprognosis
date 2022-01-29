# stdlib
from pathlib import Path
import subprocess
import sys

# adjutorium absolute
import adjutorium.logger as log

current_dir = Path(__file__).parent

predefined = {
    "combo": "git+https://github.com/yzhao062/combo",
    "hyperimpute": str(
        current_dir.parent / "third_party/hyperimpute-0.0.1-py3-none-any.whl"
    ),
    "lightgbm": "lightgbm==3.3.1",
    "line": "lime==0.2.0.1",
    "joblib": "joblib==1.1.0",
    "catboost": "catboost==1.0.3",
    "optuna": "optuna==2.10.0",
    "pycox": "pycox==0.2.2",
    "pytorch-tabnet": "pytorch-tabnet==3.1.1",
    "redis": "redis==4.1.0",
    "shap": "shap==0.40.0",
    "sklearn_pandas": "sklearn_pandas==2.2.0",
    "torch": "torch==1.9.1",
    "xgboost": "xgboost==1.5.1",
    "xgbse": "xgbse==0.2.3",
}


def install(packages: list) -> None:
    for package in packages:
        install_pack = package
        if package in predefined:
            install_pack = predefined[package]
        log.error(f"Installing {install_pack}")

        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", install_pack],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except BaseException as e:
            log.error(f"failed to install package {package}: {e}")
