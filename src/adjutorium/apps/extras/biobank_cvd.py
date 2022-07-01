# third party
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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


def extras_cbk(raw_df: pd.DataFrame, df: pd.DataFrame) -> None:
    # debug
    models = {
        "AHA/ACC score": None,
        "Framingham score": None,
        "QRisk3 score": None,
    }

    results = pd.DataFrame(
        np.zeros((1, len(models))), columns=models.keys(), index=[""]
    )
    fig = go.Figure()

    for idx, reason in enumerate(models):
        predictions = 0.12  # models[reason].predict(df, [3650]).values.squeeze()
        results[reason] = np.round(predictions, 4)
        fig.add_trace(
            go.Indicator(
                title={"text": reason, "font": {"size": STATEMENT_SIZE}},
                mode="number",
                value=np.round(predictions, 4),
                domain={"row": 0, "column": idx},
                number={"font": {"size": STATEMENT_SIZE + 3}},
            )
        )

    fig.update_layout(
        grid={"rows": 1, "columns": len(models)},
        height=80,
        margin=NO_MARGIN,
    )

    fig.update_layout(paper_bgcolor=PLOT_BACKGROUND)

    st.plotly_chart(fig)
