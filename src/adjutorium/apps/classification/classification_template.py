# stdlib
from typing import Any, Dict, List, Tuple

# third party
import pandas as pd

# adjutorium absolute
from adjutorium.apps.common.banner import banner_template
import adjutorium.apps.common.dash_reusable_components as drc
import adjutorium.logger as log
from adjutorium.plugins.explainers import Explainers
from adjutorium.studies._preprocessing import EncodersCallbacks
from adjutorium.utils.pip import install

for retry in range(2):
    try:
        # third party
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output
        import dash_bootstrap_components as dbc
        import plotly.express as px

        break
    except ImportError:
        depends = ["dash", "dash_bootstrap_components", "plotly"]
        install(depends)


external_stylesheets = [
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    dbc.themes.BOOTSTRAP,
]


def classification_dashboard(
    title: str,
    banner_title: str,
    models: Dict,
    column_types: List,
    encoders_ctx: EncodersCallbacks,
    menu_components: List,
    plot_alternatives: Dict,
) -> dash.Dash:
    """
    Dash helper for rendering the dashboard, using serialized models and menu components.

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

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    app.title = title
    CAUTION_STATEMENT = "This tool predicts your most likely outcomes based on current knowledge and data, but will never provide a 100% accurate prediction for any individual. We recommend that you discuss the results with your own specialist in a more personalised context."

    # Card components
    banner = banner_template(banner_title)

    interpretation_graphs = html.Div(
        id="interpretation_chart",
        children=[
            html.Div("Covariate impact for the risks", className="predictiontitle"),
            dcc.Graph(
                id="interpretation_chart_template", figure=px.imshow(pd.DataFrame())
            ),
            html.Br(),
        ],
    )
    workspace = html.Div(
        id="div-graphs",
        children=[
            html.Div("Risk estimation", className="predictiontitle"),
            html.Div(CAUTION_STATEMENT, className="predictionsubtitle"),
            dbc.Row(
                [
                    dbc.Col(),
                    dbc.Col(
                        dcc.Graph(
                            id="risk_chart",
                            figure=px.line(template="simple_white"),
                            style={"align": "center"},
                        )
                    ),
                    dbc.Col(),
                ]
            ),
            html.Br(),
            interpretation_graphs,
        ],
    )

    body = drc.BodyTemplate(menu_components, workspace)

    app.layout = html.Div(children=[banner, body])

    def update_interpretation(df: pd.DataFrame) -> px.imshow:
        output_figs = []
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

                if src_interpretation.shape != (1, len(df.columns)):
                    print(
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
                output_figs.extend(
                    [
                        html.Div(
                            f"Feature importance for the '{reason}' risk plot using {pretty_name}",
                            className="predictiontitle",
                        ),
                        dcc.Graph(
                            id=f"interpretation_chart_{reason}_{src}", figure=fig
                        ),
                        html.Br(),
                    ]
                )

        return output_figs

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
            title="Predictions",
            xaxis_title="Category",
            yaxis_title="Probability",
            legend_title="Categories",
        )

        return fig

    @app.callback(
        [
            Output("risk_chart", "figure"),
            Output("interpretation_chart", "children"),
        ],
        [Input(obj.name, "value") for (col, obj) in column_types],
    )
    def update_graphs(*args: Any) -> Tuple:
        for col, val in zip(column_types, args):
            if val is None:
                return px.line(template="simple_white"), []

        input_args = {}
        for k, v in zip(column_types, args):
            if isinstance(v, list):
                input_args[k[0]] = len(v)
            else:
                input_args[k[0]] = v

        raw_df = pd.DataFrame(input_args, index=[0])
        df = encoders_ctx.encode(raw_df)

        risk_chart = update_predictions(raw_df, df)
        try:
            xai_chart = update_interpretation(df)
        except BaseException as e:
            print(e)
            log.error(f"failed to generate interpretation {e}")
            return risk_chart, []
        return risk_chart, xai_chart

    return app
