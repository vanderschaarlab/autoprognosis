# stdlib
from pathlib import Path

# third party
import click

# adjutorium absolute
from adjutorium.deploy.run import start_app_server


@click.command()
@click.option("--app", type=str)
def run(app: str) -> None:
    start_app_server(Path(app))


if __name__ == "__main__":
    run()
