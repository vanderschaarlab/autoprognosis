# stdlib
import random
from pathlib import Path

# third party
import numpy as np
from sklearn.datasets import load_breast_cancer

# autoprognosis absolute
from autoprognosis.studies.classifiers import ClassifierStudy


def test_ensemble_crash() -> None:
    X, Y = load_breast_cancer(return_X_y=True, as_frame=True)

    # Simulate missingness
    total_len = len(X)

    for col in ["mean texture", "mean compactness"]:
        indices = random.sample(range(0, total_len), 10)
        X.loc[indices, col] = np.nan

    dataset = X.copy()
    dataset["target"] = Y

    workspace = Path("workspace")
    workspace.mkdir(parents=True, exist_ok=True)

    study_name = "classification_example_imputation"

    study = ClassifierStudy(
        study_name=study_name,
        dataset=dataset,
        target="target",
        num_iter=1,
        num_study_iter=1,
        timeout=1,
        imputers=["mean", "ice", "median"],
        classifiers=["logistic_regression", "lda"],
        feature_scaling=[],  # feature preprocessing is disabled
        score_threshold=0.4,
        workspace=workspace,
    )

    study.run()
