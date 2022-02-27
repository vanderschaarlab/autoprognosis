# stdlib
import argparse
from pathlib import Path

# adjutorium absolute
from adjutorium.deploy.run import start_app_server


def run(app: str, dashboard_type: str) -> None:
    start_app_server(Path(app), dashboard_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str)
    parser.add_argument("--type", type=str, default="streamlit")

    args = parser.parse_args()

    run(args.app, args.type)
