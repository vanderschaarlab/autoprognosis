# stdlib
from pathlib import Path

# adjutorium absolute
from adjutorium.deploy.run import start_app_server
from adjutorium.utils.pip import install

for retry in range(2):
    try:
        # third party
        import click

        break
    except ImportError:
        depends = ["click"]
        install(depends)


@click.command()
@click.option("--app", type=str)
def run(app: str) -> None:
    start_app_server(Path(app))


if __name__ == "__main__":
    run()
