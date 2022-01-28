# stdlib
from typing import Any, Dict, List, Tuple

# third party
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px

# adjutorium absolute
from adjutorium.apps.survival_analysis.utils.banner import banner_template
import adjutorium.apps.survival_analysis.utils.dash_reusable_components as drc
import adjutorium.logger as log
from adjutorium.plugins.explainers import Explainers
from adjutorium.studies._preprocessing import EncodersCallbacks

external_stylesheets = [
    "codepen.css",
    dbc.themes.BOOTSTRAP,
]


def survival_analysis_dashboard(
    title: str,
    banner_title: str,
    models: Dict,
    column_types: List,
    encoders_ctx: EncodersCallbacks,
    menu_components: List,
    time_horizons: List,
    plot_alternatives: Dict,
) -> dash.Dash:

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
            dcc.Graph(id="risk_chart", figure=px.line(template="simple_white")),
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

                if src_interpretation.shape != (1, len(df.columns), len(time_horizons)):
                    log.info(
                        "Interpretation source provided an invalid output {src_interpretation.shape}"
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
            log.error(f"failed to generate interpretation {e}")
            return risk_chart, []
        return risk_chart, xai_chart

    return app
