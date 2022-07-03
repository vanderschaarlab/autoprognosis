# stdlib
from typing import Any, Tuple

# third party
import numpy as np
import pandas as pd

NO_MARGIN = {
    "l": 0,  # left margin
    "r": 0,  # right margin
    "t": 0,  # top margin
    "b": 0,  # bottom margin
}
STATEMENT_SIZE = 16

PREDICTION_TEMPLATE = {
    "font": {
        "size": STATEMENT_SIZE,
    },
    "margin": NO_MARGIN,
}
PLOT_BACKGROUND = "#262730"


def extras_cbk(raw_df: pd.DataFrame, df: pd.DataFrame) -> Tuple[str, Any]:
    # debug
    models = {
        "AHA/ACC score": None,
        "Framingham score": None,
        "QRisk3 score": None,
    }

    results = pd.DataFrame(
        np.zeros((1, len(models))), columns=models.keys(), index=["10-year risk"]
    )

    for idx, reason in enumerate(models):
        predictions = 0.12  # models[reason].predict(df, [3650]).values.squeeze()
        results[reason] = np.round(predictions, 4)

    styles = [
        dict(selector="th", props=[("font-size", "18pt"), ("text-align", "center")]),
        dict(selector="tr", props=[("font-size", "16pt"), ("text-align", "center")]),
    ]

    results_styler = results.style.set_table_styles(styles)

    return ("table", results_styler)
