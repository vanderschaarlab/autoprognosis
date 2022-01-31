# stdlib
from pathlib import Path
import shutil
import subprocess

# third party
import click

# adjutorium absolute
from adjutorium.deploy.build import Builder
from adjutorium.deploy.proto import NewClassificationAppProto, NewRiskEstimationAppProto


def build_app(
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
) -> Path:
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


def build_wheel() -> Path:
    out = Path("dist")
    try:
        shutil.rmtree(out)
    except BaseException:
        pass

    subprocess.run("python setup.py bdist_wheel", shell=True, check=True)

    out_wheel = None
    for fn in out.glob("*"):
        if fn.suffix == ".whl":
            out_wheel = fn

    assert out_wheel is not None, "Invalid wheel"

    return fn


def pack(app: Path, output: Path = Path("image_bin")) -> None:
    output = Path(output)
    output_data = output / "third_party"
    try:
        shutil.rmtree(output)
    except BaseException:
        pass
    output.mkdir(parents=True, exist_ok=True)
    output_data.mkdir(parents=True, exist_ok=True)

    # Copy Adjutorium wheel
    local_wheel = build_wheel()
    shutil.copy(local_wheel, output_data / local_wheel.name)
    for fn in Path("third_party").glob("*"):
        if fn.suffix == ".whl":
            shutil.copy(fn, output_data / fn.name)

    # Copy server template
    for fn in Path("third_party/image_template").glob("*"):
        shutil.copy(fn, output / fn.name)

    # Copy server runner
    shutil.copy("scripts/run_demonstrator.py", output / "run_demonstrator.py")

    # Copy app
    shutil.copy(app, output / "app.p")

    # Update requirements txt
    with open(output / "requirements.txt", "a") as f:
        f.write(str(Path("third_party") / local_wheel.name))
        f.close()


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
@click.option("--output", type=str, default="image_bin")
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
    output: Path,
) -> None:
    app_path = build_app(
        name,
        task_type,
        dataset_path,
        model_path,
        time_column,
        target_column,
        horizons,
        explainers,
        imputers,
        plot_alternatives,
    )

    pack(app_path, output=output)


if __name__ == "__main__":
    build()
