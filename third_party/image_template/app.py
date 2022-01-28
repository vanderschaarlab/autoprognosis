# stdlib
import atexit
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from typing import Any

# third party
import psutil

WORKER_PORT = 8001
UI_PORT = 8002
N_JOBS = 1

def_path = Path("workspace").resolve()
BASE_PATH = Path(os.getenv("BASE_PATH", def_path))


def cleanup() -> None:
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for p in children:
        print("killing pid", p.pid)
        try:
            p.kill()
            p.terminate()
        except BaseException:
            continue


atexit.register(cleanup)
q: Any = queue.Queue()


def start_ui() -> None:
    local_path = Path(__file__).resolve().parent

    my_env = os.environ.copy()
    process = subprocess.Popen(
        [
            "python",
            "run_demonstrator.py",
            "--app",
            str(local_path / "app.p"),
        ],
        env=my_env,
        cwd=local_path,
    )
    process.wait()


def main(args: Any) -> None:
    start_ui()


if __name__ == "__main__":
    for retries in range(10):
        print("starting server")
        main(sys.argv[1:])
        time.sleep(1)
