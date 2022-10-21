# stdlib
from typing import Any, Dict, List

# third party
import pandas as pd

# autoprognosis absolute
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


def classification_dashboard(
    title: str,
    banner_title: str,
    models: Dict,
    column_types: List,
    encoders_ctx: EncodersCallbacks,
    menu_components: List,
    plot_alternatives: Dict,
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
        plot_alternatives: list
            List of features where to plot alternative values. Example: if treatment == 0, it will plot alternative treatment == 1 as well, as a comparison.
    """

    st.set_page_config(layout="wide", page_title=title)

    hide_footer_style = """
        <style>
        .reportview-container .main footer {visibility: hidden;}
        """
    st.markdown(hide_footer_style, unsafe_allow_html=True)
    st.markdown(
        """ <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style> """,
        unsafe_allow_html=True,
    )

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
                    label=item.name,
                )
                inputs[name] = [obj]
            if item.type == "dropdown":
                obj = st.selectbox(
                    label=item.name,
                    options=[val for val in item.val_range],
                )
                inputs[name] = [obj]
            elif item.type == "slider_integer":
                obj = st.slider(
                    item.name,
                    min_value=item.min,
                    value=item.min,
                    max_value=item.max,
                )
                inputs[name] = [obj]
            elif item.type == "slider_float":
                obj = st.slider(
                    item.name,
                    min_value=item.min,
                    value=item.min,
                    max_value=item.max,
                    step=0.1,
                )
                inputs[name] = [obj]

    def update_interpretation(df: pd.DataFrame) -> px.imshow:
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

                if src_interpretation.shape != (1, len(df.columns)):
                    log.error(
                        f"Interpretation source provided an invalid output {src_interpretation.shape}. expected {(1, len(df.columns))}"
                    )
                    continue

                interpretation_df = pd.DataFrame(
                    src_interpretation[0, :].reshape(1, -1),
                    columns=df.columns,
                    index=df.index.copy(),
                )
                interpretation_df = encoders_ctx.numeric_decode(interpretation_df)

                fig = px.imshow(
                    interpretation_df,
                    labels=dict(x="Feature", y="Source", color="Feature importance"),
                    color_continuous_scale="Blues",
                    height=250,
                )
                st.header(
                    f"Feature importance for the '{reason}' risk plot using {pretty_name}"
                )
                st.plotly_chart(fig, use_container_width=True)

    def update_predictions(raw_df: pd.DataFrame, df: pd.DataFrame) -> px.imshow:
        for reason in models:
            predictions = models[reason].predict_proba(df)
            break

        vals = {
            "Probability": predictions.values.squeeze(),
            "Category": predictions.columns,
        }
        fig = px.bar(
            vals,
            x="Category",
            y="Probability",
            color="Category",
            color_continuous_scale="RdBu",
            height=300,
            width=600,
        )
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Probability",
            legend_title="Categories",
        )

        st.header("Predictions")
        st.plotly_chart(fig, use_container_width=True)

    with predictions:
        st.header("Risk estimation")
        st.markdown(CAUTION_STATEMENT)

        raw_df = pd.DataFrame.from_dict(inputs)
        df = encoders_ctx.encode(raw_df)

        update_predictions(raw_df, df)
        update_interpretation(df)
