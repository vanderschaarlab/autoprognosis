# stdlib
from pathlib import Path

# third party
from lifelines.datasets import load_rossi

# autoprognosis absolute
from autoprognosis.deploy.build import Builder
from autoprognosis.deploy.proto import NewRiskEstimationAppProto
from autoprognosis.studies.risk_estimation import RiskEstimationStudy
from autoprognosis.utils.serialization import load_from_file


def test_surv_app() -> None:
    rossi = load_rossi()

    X = rossi.drop(["week", "arrest"], axis=1)
    Y = rossi["arrest"]
    T = rossi["week"]

    eval_time_horizons = [
        int(T[Y.iloc[:] == 1].quantile(0.50)),
        int(T[Y.iloc[:] == 1].quantile(0.75)),
    ]

    dataset = X.copy()
    dataset["target"] = Y
    dataset["time_to_event"] = T

    workspace = Path("workspace")
    workspace.mkdir(parents=True, exist_ok=True)

    study_name = "test_demonstrator_survival"

    study = RiskEstimationStudy(
        study_name=study_name,
        dataset=dataset,
        target="target",
        time_to_event="time_to_event",
        time_horizons=eval_time_horizons,
        num_iter=2,
        num_study_iter=1,
        timeout=60,
        risk_estimators=["cox_ph"],
        imputers=["mean"],
        feature_scaling=["minmax_scaler", "nop"],
        score_threshold=0.4,
        workspace=workspace,
    )

    study.run()

    dataset_path = workspace / "demo_dataset_surv.csv"
    dataset.to_csv(dataset_path, index=None)

    name = "AutoPrognosis demo: Survival Analysis"
    model_path = workspace / study_name / "model.p"

    time_column = "time_to_event"
    target_column = "target"
    task_type = "risk_estimation"

    task = Builder(
        NewRiskEstimationAppProto(
            **{
                "name": name,
                "type": task_type,
                "dataset_path": str(dataset_path),
                "model_path": str(model_path),
                "time_column": time_column,
                "target_column": target_column,
                "horizons": eval_time_horizons,
                "explainers": ["kernel_shap"],
                "imputers": [],
                "plot_alternatives": [],
                "comparative_models": [
                    (
                        "Cox PH",  # display name
                        "cox_ph",  # autoprognosis plugin name
                        {},  # plugin args
                    ),
                ],
                "extras_cbk": None,
                "auth": False,
                "extras_cbk": None,
            }
        ),
    )

    app_path = task.run()
    app = load_from_file(app_path)

    assert app["title"] == name
    assert app["type"] == "risk_estimation"
    assert app["banner_title"] == f"{name} study"
    assert len(app["models"]) > 0
    assert "encoders" in app
    assert "menu_components" in app
    assert "column_types" in app
