# third party
import click

# adjutorium absolute
from adjutorium.deploy.build import Builder
from adjutorium.deploy.proto import NewClassificationAppProto, NewRiskEstimationAppProto


@click.command()
@click.option("--name", type=str, default="new_demonstrator")
@click.option("--task_type", type=str)
@click.option("--dataset_path", type=str)
@click.option("--model_path", type=str)
@click.option("--time_column", type=str)
@click.option("--target_column", type=str)
@click.option("--horizons", type=str)
@click.option("--explainers", type=str, default="kernel_shap")
@click.option("--imputers", type=str, default="ice")
@click.option("--plot_alternatives", type=str, default=[])
def build(
    name: str,
    task_type: str,
    dataset_path: str,
    model_path: str,
    time_column: str,
    target_column: str,
    horizons: str,
    explainers: str,
    imputers: str,
    plot_alternatives: str,
) -> str:
    print(task_type)
    if task_type == "risk_estimation":
        parsed_horizons = []
        for tok in horizons.split(","):
            parsed_horizons.append(int(tok))

        task = Builder(
            NewRiskEstimationAppProto(
                **{
                    "name": name,
                    "type": task_type,
                    "dataset_path": dataset_path,
                    "model_path": model_path,
                    "time_column": time_column,
                    "target_column": target_column,
                    "horizons": parsed_horizons,
                    "explainers": explainers.split(","),
                    "imputers": imputers.split(","),
                    "plot_alternatives": [],
                }
            )
        )
    elif task_type == "classification":
        task = Builder(
            NewClassificationAppProto(
                **{
                    "name": name,
                    "type": task_type,
                    "dataset_path": dataset_path,
                    "model_path": model_path,
                    "target_column": target_column,
                    "explainers": explainers.split(","),
                    "imputers": imputers.split(","),
                    "plot_alternatives": [],
                }
            )
        )
    else:
        raise RuntimeError(f"unsupported type {type}")

    return task.run()


if __name__ == "__main__":
    build()
