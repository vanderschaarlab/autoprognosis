# third party
import pandas as pd

# autoprognosis absolute
from autoprognosis.plugins.prediction import Predictions


def test_train_error() -> None:
    model = Predictions().get("logistic_regression")

    err = ""
    try:
        model.predict_proba(pd.DataFrame([]))
    except BaseException as e:
        err = str(e)

    assert "Fit the model first" == err
