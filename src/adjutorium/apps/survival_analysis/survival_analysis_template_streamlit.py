# stdlib
from typing import Any, Callable, Dict, List

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


LOGO_URL = "https://www.vanderschaar-lab.com/wp-content/uploads/2020/04/transpLogo_long_plus.png"
SITE_URL = "https://www.vanderschaar-lab.com/"
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
CAUTION_STATEMENT = "This tool predicts your most likely outcomes based on current knowledge and data, but will never provide a 100% accurate prediction for any individual. We recommend that you discuss the results with your own specialist in a more personalised context."

PLOT_STATEMENT = "The following graphs show diagnostic risk predictions up to 10 years, using Adjutorium and other benchmark models."
PLOT_BACKGROUND = "#262730"


def survival_analysis_dashboard(
    title: str,
    banner_title: str,
    models: Dict,
    column_types: List,
    encoders_ctx: EncodersCallbacks,
    menu_components: List,
    time_horizons: List,
    plot_alternatives: Dict,
    extras_cbk: Callable = None,
) -> Any:
    """
    Streamlit helper for rendering the dashboard, using serialized models and menu components.

    Args:
        title:
            Page title
        banner_title:
            The title used in the banner.
        models:
            The models used for evaluation and plots.
        column_types: List
            List of the dataset features and their distribution.
        encoders_ctx: EncodersCallbacks,
            List of encoders/decoders for the menu values < - > model input values.
        menu_components: List
            Type of menu item for each feature: checkbox, dropdown etc.
        time_horizons: list
            List of horizons to plot.
        plot_alternatives: list
            List of features where to plot alternative values. Example: if treatment == 0, it will plot alternative treatment == 1 as well, as a comparison.
    """
    st.set_page_config(layout="wide", page_title=title)

    base_style = """
        <style>
        #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 3rem;}
        .reportview-container .main footer {visibility: hidden;}
        </style>
        """
    st.markdown(
        base_style,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            f"""
            <div style="float:left;"><h1>{title}</h1></div>
            <div style="float:right;">
                <a href="{SITE_URL}">
                    <img align='left' height='70px' src='{LOGO_URL}'/>
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

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
                    value=int(item.median),
                    max_value=int(item.max),
                    step=1,
                )
                inputs[name] = [obj]
            elif item.type == "slider_float":
                obj = st.slider(
                    item.name,
                    min_value=float(item.min),
                    value=float(item.median),
                    max_value=float(item.max),
                    step=0.1,
                )
                inputs[name] = [obj]

    def update_interpretation(df: pd.DataFrame) -> None:
        figs = []
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

                if src_interpretation.shape == (1, len(df.columns), len(time_horizons)):
                    display_interpretation = []

                    for idx, h in enumerate(time_horizons):
                        interpretation_df = pd.DataFrame(
                            src_interpretation[0, :, idx].reshape(1, -1),
                            columns=df.columns,
                            index=df.index.copy(),
                        )
                        interpretation_df = encoders_ctx.numeric_decode(
                            interpretation_df
                        )
                        display_interpretation = pd.concat(
                            [display_interpretation, interpretation_df]
                        )

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
                    fig.update_layout(
                        **PREDICTION_TEMPLATE,
                    )
                    figs.append(
                        (
                            f"Feature importance for the '{reason}' risk plot using {pretty_name}",
                            fig,
                        )
                    )
                elif src_interpretation.shape == (1, len(df.columns)):
                    interpretation_df = pd.DataFrame(
                        src_interpretation[0, :].reshape(1, -1),
                        columns=df.columns,
                        index=df.index.copy(),
                    )

                    interpretation_df = encoders_ctx.numeric_decode(interpretation_df)
                    interpretation_df = interpretation_df.sort_values(
                        by=list(interpretation_df.index)[0], axis=1, ascending=False
                    )
                    cols = interpretation_df.columns
                    if len(cols) > 10:
                        cols = cols[:10]

                    fig = px.imshow(
                        interpretation_df[cols],
                        labels=dict(
                            x="Feature", y="Importance", color="Feature importance"
                        ),
                        color_continuous_scale="OrRd",
                        height=190,
                    )

                    fig.update_layout(
                        **PREDICTION_TEMPLATE,
                    )

                    figs.append(
                        (
                            f"Feature importance for the '{reason}' risk plot using {pretty_name}",
                            fig,
                        )
                    )

                else:
                    print(
                        f"Interpretation source provided an invalid output {src_interpretation.shape}. expected {(1, len(df.columns), len(time_horizons))}"
                    )
                    continue
        return figs

    def update_predictions(raw_df: pd.DataFrame, df: pd.DataFrame) -> None:
        print("predict", df)
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
                    "alternative": "",
                    "risk": risk_estimation,
                    # "uncertainity": uncertainity_estimation,
                    "years": time_horizons,
                    "reason": [reason] * len(time_horizons),
                }
            )
            output_df = pd.concat([output_df, local_output_df])

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
                    output_df = pd.concat([output_df, alt_output_df])

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
                "risk": "CVD Risk probability [0 - 1]",
                "reason": "Risk",
                "alternative": "Scenario",
            },
            template="plotly_dark",
            color_discrete_sequence=[
                "#e41a1c",
                "#ff7f00",
                "#377eb8",
                "#4daf4a",
                "#984ea3",
                "#a65628",
                "#f781bf",
            ],
            markers=True,
        )
        fig.update_layout(
            legend_title="Prediction Model",
            hovermode="x unified",
            **PREDICTION_TEMPLATE,
        )
        fig.update_traces(line=dict(width=3))

        return fig

    with predictions:
        gif_runner = st.image("assets/loading.gif")

        raw_df = pd.DataFrame.from_dict(inputs)
        df = encoders_ctx.encode(raw_df)

        prediction_fig = update_predictions(raw_df, df)
        extras_type, extras_data = None, None
        if extras_cbk is not None:
            extras_type, extras_data = extras_cbk(raw_df, df)
        xai_figs = update_interpretation(df)

        gif_runner.empty()

        # Title
        st.header("Risk estimation")
        st.markdown(
            f'<p style="font-size: {STATEMENT_SIZE}px;">' + CAUTION_STATEMENT + "</p>",
            unsafe_allow_html=True,
        )

        # Charts
        st.subheader("Predictions")

        st.markdown(
            f'<p style="font-size: {STATEMENT_SIZE}px;">' + PLOT_STATEMENT + "</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(prediction_fig, use_container_width=True)

        # Other benchmarks
        st.table(extras_data)

        # XAI data
        for xai_title, xai_fig in xai_figs:
            st.subheader(xai_title)

            st.plotly_chart(xai_fig, use_container_width=True)
