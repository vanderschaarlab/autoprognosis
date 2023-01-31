# stdlib
from pathlib import Path

# third party
import numpy as np
from sklearn.datasets import load_iris

# autoprognosis absolute
from autoprognosis.deploy.build import Builder
from autoprognosis.deploy.proto import NewClassificationAppProto
from autoprognosis.studies.classifiers import ClassifierStudy
from autoprognosis.utils.serialization import load_from_file


def test_sanity():
    X, Y = load_iris(return_X_y=True, as_frame=True)

    df = X.copy()
    df["target"] = Y

    df.loc[:2, "sepal length (cm)"] = np.nan

    workspace = Path("workspace")
    workspace.mkdir(parents=True, exist_ok=True)

    study_name = "test_demonstrator_classification"

    study = ClassifierStudy(
        study_name=study_name,
        dataset=df,  # pandas DataFrame
        target="target",  # the label column in the dataset
        timeout=60,  # timeout for optimization for each classfier. Default: 600 seconds
        num_iter=5,
        num_study_iter=1,
        classifiers=["logistic_regression"],
        workspace=workspace,
    )

    study.run()

    dataset_path = workspace / "demo_dataset_classification.csv"
    df.to_csv(dataset_path, index=None)

    name = "AutoPrognosis demo: Classification"
    model_path = workspace / study_name / "model.p"

    target_column = "target"
    task_type = "classification"

    task = Builder(
        NewClassificationAppProto(
            **{
                "name": name,
                "type": task_type,
                "dataset_path": str(dataset_path),
                "model_path": str(model_path),
                "target_column": target_column,
                "explainers": ["kernel_shap"],
                "imputers": [],
                "plot_alternatives": [],
                "comparative_models": [
                    (
                        "Logistic regression",  # display name
                        "logistic_regression",  # autoprognosis plugin name
                        {},  # plugin args
                    ),
                ],
                "auth": False,
            }
        ),
    )

    app_path = task.run()

    app = load_from_file(app_path)

    assert app["title"] == name
    assert app["type"] == "classification"
    assert app["banner_title"] == f"{name} study"
    assert len(app["models"]) > 0
    assert "encoders" in app
    assert "menu_components" in app
    assert "column_types" in app
