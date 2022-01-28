# stdlib
from typing import Any, Dict

# third party
from dash import dcc, html
import dash_bootstrap_components as dbc

menu_font_size = "14px"
menu_font_family = "Gill Sans, Gill Sans MT, Calibri, sans-serif"


def _omit(omitted_keys: Any, d: Dict) -> Dict:
    return {k: v for k, v in d.items() if k not in omitted_keys}


# Custom Display Components
def Card(children: Any, **kwargs: Any) -> html.Section:
    return html.Section(className="card", children=children, **_omit(["style"], kwargs))


def InfoCard(info: str, subinfo: str, **kwargs: Any) -> dbc.Card:
    return dbc.Card(
        className="infocard",
        id=subinfo,
        children=[
            html.H5(info, className="card-title"),
            html.P(subinfo, className="card-text"),
        ],
        **_omit(["style"], kwargs),
    )


def NamedInput(name: str, step: int = 1, **kwargs: Any) -> html.Div:
    result = html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[
            html.P(
                f"{name}:",
                style={"font-size": menu_font_size, "font-family": menu_font_family},
            ),
            html.Div(
                children=[
                    dcc.Input(
                        className="inputinteger",
                        placeholder="---",
                        debounce=True,
                        type="number",
                        step=step,
                        id=name,
                        style={
                            "font-size": menu_font_size,
                            "font-family": menu_font_family,
                        },
                        **kwargs,
                    ),
                ],
            ),
        ],
    )

    return result


def NamedDropdown(name: str, **kwargs: Any) -> html.Div:
    return html.Div(
        style={
            "margin": "10px 0px",
            "font-size": menu_font_size,
            "font-family": menu_font_family,
        },
        children=[
            html.P(
                children=f"{name}:",
                style={
                    "margin-left": "3px",
                    "font-size": menu_font_size,
                    "font-family": menu_font_family,
                },
            ),
            dcc.Dropdown(
                id=name,
                placeholder="Select value",
                clearable=False,
                searchable=True,
                style={
                    "font-size": menu_font_size,
                    "font-family": menu_font_family,
                },
                **kwargs,
            ),
        ],
    )


def NamedCheckbox(name: str, **kwargs: Any) -> html.Div:
    return html.Div(
        style={"margin": "10px 0px"},
        children=[
            dcc.Checklist(
                options=[
                    {"label": name, "value": 1},
                ],
                value=[],
                id=name,
                labelStyle={
                    "font-size": menu_font_size,
                    "font-family": menu_font_family,
                },
                inputStyle={
                    "font-size": menu_font_size,
                    "font-family": menu_font_family,
                    "margin-right": "2px",
                    "margin-left": "5px",
                },
                **kwargs,
            ),
        ],
    )


def BodyTemplate(menu_components: Any, workspace: Any) -> html.Div:
    return html.Div(
        id="body",
        className="container scalable",
        children=[
            html.Div(
                id="app-container",
                children=[
                    html.Div(
                        className="leftmenu",
                        id="left-column",
                        children=[
                            Card(
                                id="menu-card",
                                children=menu_components,
                                style={
                                    "font-size": menu_font_size,
                                    "font-family": menu_font_family,
                                },
                            ),
                        ],
                    ),
                    workspace,
                ],
            )
        ],
    )
