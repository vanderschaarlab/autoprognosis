# stdlib
from typing import Any, Dict, List

# third party
import numpy as np
import pandas as pd

# adjutorium absolute
from adjutorium.plugins.explainers import Explainers
from adjutorium.studies._preprocessing import EncodersCallbacks
from adjutorium.utils.pip import install

for retry in range(2):
    try:
        # third party
        import plotly.express as px
        import streamlit as st

        break
    except ImportError:
        depends = ["streamlit", "plotly"]
        install(depends)


def survival_analysis_dashboard(
    title: str,
    banner_title: str,
    models: Dict,
    column_types: List,
    encoders_ctx: EncodersCallbacks,
    menu_components: List,
    time_horizons: List,
    plot_alternatives: Dict,
) -> Any:
    print(title)
    st.set_page_config(layout="wide", page_title=title)

    hide_footer_style = """
        <style>
        .reportview-container .main footer {visibility: hidden;}
        """
    st.markdown(hide_footer_style, unsafe_allow_html=True)
    # st.markdown(
    #    """ <style>
    #    #MainMenu {visibility: hidden;}
    #    footer {visibility: hidden;}
    # </style> """,
    #    unsafe_allow_html=True,
    # )

    with st.container():
        st.markdown(
            f"<h1 style='margin-top: -70px;'>{title}</h1>", unsafe_allow_html=True
        )
        st.markdown("---")

    CAUTION_STATEMENT = "This tool predicts your most likely outcomes based on current knowledge and data, but will never provide a 100% accurate prediction for any individual. We recommend that you discuss the results with your own specialist in a more personalised context."

    menu, predictions = st.columns([1, 4])

    inputs = {}
    columns = []
    with menu:
        st.header("Patient info")
        for name, item in menu_components:
            columns.append(name)
            if item.type == "checkbox":
                obj = st.checkbox(
                    item.name,
                )
                inputs[name] = [obj]
            if item.type == "dropdown":
                obj = st.selectbox(
                    item.name,
                    options=item.val_range,
                )
                inputs[name] = [obj]
            elif item.type == "slider_integer":
                obj = st.slider(
                    item.name,
                    min_value=int(item.min),
                    value=int(item.min),
                    max_value=int(item.max),
                    step=1,
                )
                inputs[name] = [obj]
            elif item.type == "slider_float":
                obj = st.slider(
                    item.name,
                    min_value=float(item.min),
                    value=float(item.min),
                    max_value=float(item.max),
                    step=0.1,
                )
                inputs[name] = [obj]

    def update_interpretation(df: pd.DataFrame) -> None:
        for reason in models:
            if not hasattr(models[reason], "explain"):
                continue
            try:
                raw_interpretation = models[reason].explain(df)
                assert isinstance(raw_interpretation, dict)
            except BaseException:
                continue

            for src in raw_interpretation:
                pretty_name = Explainers().get_type(src).pretty_name()
                src_interpretation = raw_interpretation[src]

                if src_interpretation.shape != (1, len(df.columns), len(time_horizons)):
                    print(
                        f"Interpretation source provided an invalid output {src_interpretation.shape}. expected {(1, len(df.columns), len(time_horizons))}"
                    )
                    continue

                display_interpretation = []

                for idx, h in enumerate(time_horizons):
                    interpretation_df = pd.DataFrame(
                        src_interpretation[0, :, idx].reshape(1, -1),
                        columns=df.columns,
                        index=df.index.copy(),
                    )
                    interpretation_df = encoders_ctx.numeric_decode(interpretation_df)
                    display_interpretation.append(interpretation_df.values)
                interpretation = np.asarray(display_interpretation).T.squeeze()
                interpretation = (interpretation - interpretation.min()) / (
                    interpretation.max() - interpretation.min() + 1e-8
                )

                fig = px.imshow(
                    interpretation,
                    y=interpretation_df.columns,
                    x=np.asarray(time_horizons) / 365,
                    labels=dict(x="Years", y="Feature", color="Feature importance"),
                    color_continuous_scale="OrRd",
                )
                st.header(
                    f"Feature importance for the '{reason}' risk plot using {pretty_name}"
                )
                st.plotly_chart(fig, use_container_width=True)

    def update_predictions(raw_df: pd.DataFrame, df: pd.DataFrame) -> None:
        output_df = pd.DataFrame(
            {
                "alternative": [],
                "risk": [],
                "years": [],
                "reason": [],
            }
        )

        for reason in models:
            predictions = models[reason].predict(df, time_horizons)
            risk_estimation = predictions.values[0]
            # uncertainity_estimation = uncertainity.values[0]

            local_output_df = pd.DataFrame(
                {
                    "alternative": "(for the selected parameters)",
                    "risk": risk_estimation,
                    # "uncertainity": uncertainity_estimation,
                    "years": time_horizons,
                    "reason": [reason] * len(time_horizons),
                }
            )
            output_df = output_df.append(local_output_df)

        for reason in models:
            if reason not in plot_alternatives:
                continue

            for col in plot_alternatives[reason]:

                if col not in raw_df.columns:
                    print("unknown column provided", col)
                    continue

                current_val = raw_df[col].values[0]
                for alternative_val in plot_alternatives[reason][col]:
                    if alternative_val == current_val:
                        continue

                    alt_raw_df = raw_df.copy()
                    alt_raw_df[col] = alternative_val

                    try:
                        alt_df = encoders_ctx.encode(alt_raw_df)
                    except BaseException as e:
                        print("failed to encode", str(e))
                        continue
                    predictions = models[reason].predict(alt_df, time_horizons)
                    alt_risk_estimation = predictions.values[0]

                    alt_output_df = pd.DataFrame(
                        {
                            "alternative": "(alternative)",
                            "risk": alt_risk_estimation,
                            "years": time_horizons,
                            "reason": [f"{reason} with : {col} = {alternative_val}"]
                            * len(time_horizons),
                        }
                    )
                    output_df = output_df.append(alt_output_df)

        output_df["risk"] *= 100
        output_df["years"] /= 365

        fig = px.line(
            output_df,
            x="years",
            y="risk",
            # error_y="uncertainity",
            color="reason",
            line_dash="alternative",
            labels={
                "years": "Years to event",
                "risk": "Risk probability",
                "reason": "Risk",
                "alternative": "Scenario",
            },
            template="simple_white",
            title="Risk prediction",
        )

        st.header("Predictions")
        st.plotly_chart(fig, use_container_width=True)

    with predictions:
        print(inputs)
        st.header("Risk estimation")
        st.markdown(CAUTION_STATEMENT)

        raw_df = pd.DataFrame.from_dict(inputs)
        df = encoders_ctx.encode(raw_df)

        update_predictions(raw_df, df)
        update_interpretation(df)
