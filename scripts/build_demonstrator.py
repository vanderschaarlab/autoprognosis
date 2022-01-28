# third party
import click

# adjutorium absolute
from adjutorium.deploy.build import Builder
from adjutorium.deploy.proto import NewAppProto


@click.command()
@click.option("--name", type=str, default="new_demonstrator")
@click.option("--type", type=str, default="risk_estimation")
@click.option("--dataset_path", type=str)
@click.option("--model_path", type=str)
@click.option("--time_column", type=str)
@click.option("--target_column", type=str)
@click.option("--horizons", type=float, multiple=True)
@click.option("--explainers", type=str, multiple=True, default=[])
@click.option("--imputers", type=str, multiple=True, default=["ice"])
@click.option("--plot_alternatives", type=str, multiple=True, default=[])
def build(
    name: str,
    type: str,
    dataset_path: str,
    model_path: str,
    time_column: str,
    target_column: str,
    horizons: list,
    explainers: list,
    imputers: list,
    plot_alternatives: list,
) -> None:
    print(horizons)
    task = Builder(
        NewAppProto(
            **{
                "name": name,
                "type": type,
                "dataset_path": dataset_path,
                "model_path": model_path,
                "time_column": time_column,
                "target_column": target_column,
                "horizons": horizons,
                "explainers": explainers,
                "imputers": imputers,
                "plot_alternatives": plot_alternatives,
            }
        )
    )

    task.run()

    print(task.status)


if __name__ == "__main__":
    build()
