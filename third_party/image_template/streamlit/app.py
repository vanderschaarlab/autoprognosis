# stdlib
import subprocess
import sys


def install(install_pack: str) -> None:
    print(f"Installing {install_pack}")

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", install_pack],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    # install("third_party/autoprognosis-0.1.1-py2.py3-none-any.whl")
    # third party
    from run_demonstrator import run

    run("app.p")
