# stdlib
from typing import Any, Callable, Dict, List

# third party
import numpy as np
import pandas as pd

# autoprognosis absolute
from autoprognosis.apps.common.login import (
    clean_blocks,
    generate_login_block,
    is_authenticated,
    login,
)
import autoprognosis.logger as log
from autoprognosis.plugins.explainers import Explainers
from autoprognosis.studies._preprocessing import EncodersCallbacks
from autoprognosis.utils.pip import install

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

PLOT_STATEMENT = "The following graphs show diagnostic risk predictions up to 12 years, using AutoPrognosis and other benchmark models."
PLOT_BACKGROUND = "#262730"


def generate_page_config(title):
    st.set_page_config(layout="wide", page_title=title)

    base_style = """
        <style>
        #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 3rem;}
        .reportview-container .main footer {visibility: hidden;}
        div.stButton > button:first-child {
            font-size:20px;
        }

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


def generate_menu_items(menu_components):
    inputs = {}
    columns = []

    with st.form("patient_form"):
        st.header("Patient info")
        submitted = st.form_submit_button(" Evaluate Risk  ▶️  ")

        for idx, (name, item) in enumerate(menu_components):
            if item.type == "header":
                st.markdown("---")
                st.markdown("##### " + item.name)
                continue
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

    return submitted, inputs, columns


def generate_interpretation_plots(
    models: list, df: pd.DataFrame, time_horizons: list, encoders_ctx
) -> None:
    figs = []
    for reason in models:
        if not hasattr(models[reason], "explain"):
            continue
        try:
            raw_interpretation = models[reason].explain(df)
            if not isinstance(raw_interpretation, dict):
                raise ValueError("raw_interpretation must be a dict")
        except BaseException:
            continue

        for src in raw_interpretation:
            pretty_name = Explainers().get_type(src).pretty_name()
            src_interpretation = raw_interpretation[src]

            if src_interpretation.shape == (1, len(df.columns), len(time_horizons)):
                src_interpretation = np.mean(src_interpretation, axis=2)

            if src_interpretation.squeeze().shape == (len(df.columns),):
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
                    height=100,
                )
                fig.update_layout(
                    **PREDICTION_TEMPLATE,
                )
                fig.update_layout(showlegend=False)

                figs.append(
                    (
                        f"Feature importance for the '{reason}' risk plot using {pretty_name}",
                        fig,
                    )
                )

            else:
                log.error(
                    f"Interpretation source provided an invalid output {src_interpretation.shape}. expected {(1, len(df.columns), len(time_horizons))}"
                )
                continue
    return figs


def generate_predictions_plots(
    models: list,
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    time_horizons: list,
    encoders_ctx: EncodersCallbacks,
    plot_alternatives: Dict,
) -> None:
    output_df = pd.DataFrame(
        {
            "alternative": [],
            "risk": [],
            "years": [],
            "reason": [],
        }
    )

    for reason in models:
        predictions, uncertainity_estimation = models[reason].predict_with_uncertainty(
            df, time_horizons
        )
        risk_estimation = predictions.values[0]
        uncertainity_estimation = np.asarray(uncertainity_estimation).squeeze()

        local_output_df = pd.DataFrame(
            {
                "risk": risk_estimation,
                "uncertainity": uncertainity_estimation,
                "years": time_horizons,
                "reason": [reason] * len(time_horizons),
            }
        )
        output_df = pd.concat([output_df, local_output_df])

    output_df["years"] /= 365

    fig = px.line(
        output_df,
        x="years",
        y="risk",
        error_y="uncertainity",
        color="reason",
        labels={
            "years": "Years to event",
            "risk": "Risk probability [0 - 1]",
            "reason": "Model",
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


def generate_survival_analysis_dashboard(
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

    menu, predictions = st.columns([1, 4])

    inputs = {}
    columns = []

    with menu:
        submitted, inputs, columns = generate_menu_items(menu_components)

    with predictions:
        if submitted:
            # Title
            st.header("Risk estimation")
            st.markdown(
                f'<p style="font-size: {STATEMENT_SIZE}px;">'
                + CAUTION_STATEMENT
                + "</p>",
                unsafe_allow_html=True,
            )

            # Charts
            with st.spinner("Generating predictions..."):
                raw_df = pd.DataFrame.from_dict(inputs)
                df = encoders_ctx.encode(raw_df)

                prediction_fig = generate_predictions_plots(
                    models, raw_df, df, time_horizons, encoders_ctx, plot_alternatives
                )

                st.subheader("Predictions")

                st.markdown(
                    f'<p style="font-size: {STATEMENT_SIZE}px;">'
                    + PLOT_STATEMENT
                    + "</p>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(prediction_fig, use_container_width=True)

            # Other benchmarks
            with st.spinner("Evaluating reference models..."):
                extras_type, extras_data = None, None
                if extras_cbk is not None:
                    extras_type, extras_data = extras_cbk(raw_df)

                    st.table(extras_data)

            # XAI data
            with st.spinner("Evaluating feature importance..."):
                xai_figs = generate_interpretation_plots(
                    models, df, time_horizons, encoders_ctx
                )
                for xai_title, xai_fig in xai_figs:
                    st.subheader(xai_title)

                    st.plotly_chart(xai_fig, use_container_width=True)


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
    auth: bool = False,
) -> Any:
    generate_page_config(title)

    if auth is False:
        return generate_survival_analysis_dashboard(
            title=title,
            banner_title=banner_title,
            models=models,
            column_types=column_types,
            encoders_ctx=encoders_ctx,
            menu_components=menu_components,
            time_horizons=time_horizons,
            plot_alternatives=plot_alternatives,
            extras_cbk=extras_cbk,
        )

    login_key = "login_state"

    if login_key not in st.session_state:
        st.session_state[login_key] = False

    prev_login = st.session_state[login_key]
    if not auth or prev_login:
        return generate_survival_analysis_dashboard(
            title=title,
            banner_title=banner_title,
            models=models,
            column_types=column_types,
            encoders_ctx=encoders_ctx,
            menu_components=menu_components,
            time_horizons=time_horizons,
            plot_alternatives=plot_alternatives,
            extras_cbk=extras_cbk,
        )

    login_blocks = generate_login_block()
    password = login(login_blocks)

    if is_authenticated(password):
        clean_blocks(login_blocks)
        st.session_state[login_key] = True
        generate_survival_analysis_dashboard(
            title=title,
            banner_title=banner_title,
            models=models,
            column_types=column_types,
            encoders_ctx=encoders_ctx,
            menu_components=menu_components,
            time_horizons=time_horizons,
            plot_alternatives=plot_alternatives,
            extras_cbk=extras_cbk,
        )
    elif password:
        st.info("Please enter a valid password")
