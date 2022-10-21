# stdlib
import argparse
from pathlib import Path

# autoprognosis absolute
from autoprognosis.deploy.run import start_app_server


def run(app: str) -> None:
    start_app_server(Path(app))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str)

    args = parser.parse_args()

    run(args.app)
