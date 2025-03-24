# stdlib
import shutil
import subprocess
from pathlib import Path

# third party
import click

# autoprognosis absolute
from autoprognosis.apps.extras.biobank_cvd import extras_cbk as biobank_cvd_extras_cbk
from autoprognosis.apps.extras.biobank_diabetes import (
    extras_cbk as biobank_diabetes_extras_cbk,
)
from autoprognosis.deploy.build import Builder
from autoprognosis.deploy.proto import (
    NewClassificationAppProto,
    NewRiskEstimationAppProto,
)


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
    extras: str,
    auth: bool,
) -> Path:
    def split_and_clean(raw: str) -> list:
        lst = raw.split(",")
        if "" in lst:
            lst.remove("")

        return lst

    extras_cbk = None
    if extras == "biobank_cvd":
        extras_cbk = biobank_cvd_extras_cbk
    if extras == "biobank_diabetes":
        extras_cbk = biobank_diabetes_extras_cbk

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
                    "explainers": split_and_clean(explainers),
                    "imputers": split_and_clean(imputers),
                    "plot_alternatives": [],
                    "comparative_models": [
                        (
                            "Cox PH",
                            "cox_ph",
                            {
                                "alpha": 0.014721404833448894,
                                "penalizer": 0.08157265024269905,
                            },
                        ),
                        (
                            "Survival XGB",
                            "survival_xgboost",
                            {
                                "max_depth": 2,
                                "min_child_weight": 15,
                                "objective": "cox",
                                "strategy": "weibull",
                            },
                        ),
                    ],
                    "extras_cbk": extras_cbk,
                    "auth": auth,
                }
            ),
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
                    "explainers": split_and_clean(explainers),
                    "imputers": split_and_clean(imputers),
                    "plot_alternatives": [],
                }
            )
        )
    else:
        raise RuntimeError(f"unsupported type {type}")

    return Path(task.run())


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


def pack(
    app: Path,
    output: Path = Path("output/image_bin"),
) -> None:
    output = Path(output)
    output_data = output / "third_party"
    try:
        shutil.rmtree(output)
    except BaseException:
        pass
    output.mkdir(parents=True, exist_ok=True)
    output_data.mkdir(parents=True, exist_ok=True)

    # Copy AutoPrognosis wheel
    local_wheel = build_wheel()
    shutil.copy(local_wheel, output_data / local_wheel.name)
    for fn in Path("third_party").glob("*"):
        if fn.suffix == ".whl":
            shutil.copy(fn, output_data / fn.name)

    # Copy server template
    for fn in Path("third_party/image_template/streamlit").glob("*"):
        if Path(fn).is_file():
            shutil.copy(fn, output / fn.name)
        else:
            shutil.copytree(fn, output / fn.name)

    # Copy server runner
    shutil.copy("scripts/run_demonstrator.py", output / "run_demonstrator.py")

    # Copy app
    shutil.copy(app, output / "app.p")


@click.command()
@click.option(
    "--name", type=str, default="new_demonstrator", help="The title of the demonstrator"
)
@click.option("--task_type", type=str, help="classification/risk_estimation")
@click.option("--dataset_path", type=str, help="Path to the dataset csv")
@click.option(
    "--model_path", type=str, help="Path to the model template, usually model.p"
)
@click.option(
    "--time_column",
    type=str,
    help="Only for risk_estimation tasks. Which column in the dataset is used for time-to-event",
)
@click.option(
    "--target_column", type=str, help="Which column in the dataset is the outcome"
)
@click.option(
    "--horizons",
    type=str,
    help="Only for risk_estimation tasks. Which time horizons to plot.",
)
@click.option(
    "--explainers",
    type=str,
    default="kernel_shap",
    help="Which explainers to include. There can be multiple explainer names, separated by a comma. Available explainers: kernel_shap,invase,shap_permutation_sampler,lime.",
)
@click.option(
    "--imputers",
    type=str,
    default="ice",
    help="Which imputer to use. Available imputers: ['sinkhorn', 'EM', 'mice', 'ice', 'hyperimpute', 'most_frequent', 'median', 'missforest', 'softimpute', 'nop', 'mean', 'gain']",
)
@click.option(
    "--plot_alternatives",
    type=str,
    default=[],
    help="Only for risk_estimation. List of categorical columns by which to split the graphs. For example, plot outcome for different treatments available.",
)
@click.option(
    "--extras",
    type=str,
    default="",
    help="Task specific callback, like biobank_cvd or biobank_diabetes.",
)
@click.option(
    "--output",
    type=str,
    default="output",
    help="Where to save the demonstrator files. The content of the folder can be directly used for deployments(for example, to Heroku).",
)
@click.option(
    "--auth",
    type=bool,
    default=False,
    help="Optional. If provided, the dashboard will be protected by a password.",
)
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
    extras: str,
    output: Path,
    auth: bool,
) -> None:
    output = Path(output)
    try:
        shutil.rmtree(output)
    except BaseException:
        pass
    output.mkdir(parents=True, exist_ok=True)

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
        extras,
        auth=auth,
    )

    image_bin = Path(output) / "image_bin"
    pack(app_path, output=image_bin)


if __name__ == "__main__":
    build()
