# adjutorium absolute
from adjutorium.apps.common.assets.styles import (
    page_banner_body,
    page_banner_border,
    page_banner_button_right_style,
    page_banner_button_style,
    page_banner_image_style,
    page_banner_link_title_style,
    page_banner_title_style,
)
from adjutorium.utils.pip import install

for retry in range(2):
    try:
        # third party
        from dash import html
        import dash_bootstrap_components as dbc

        break
    except ImportError:
        depends = ["dash", "dash_bootstrap_components"]
        install(depends)


def banner_template(title: str) -> html.Div:
    """Helper for generating the banner for the page."""
    banner_content = dbc.Row(
        [
            dbc.Col(
                html.H2(
                    children=[
                        html.A(
                            title,
                            href="/",
                            style=page_banner_link_title_style,
                        )
                    ],
                    style=page_banner_title_style,
                )
            ),
            dbc.Col(
                html.A(
                    dbc.Button("Send Feedback"),
                    href="https://www.vanderschaar-lab.com/contact-us/",
                    target="_blank",
                ),
                width=1,
                style=page_banner_button_style,
            ),
            dbc.Col(
                html.A(
                    dbc.Button("Learn more"),
                    href="https://www.vanderschaar-lab.com/",
                    target="_blank",
                ),
                width=1,
                style=page_banner_button_right_style,
            ),
            dbc.Col(
                html.A(
                    children=[
                        html.Img(
                            src="https://www.vanderschaar-lab.com/wp-content/uploads/2020/04/transpLogo_long_plus.png",
                            style={
                                "height": "60px",
                            },
                        )
                    ],
                    href="https://www.vanderschaar-lab.com/",
                    target="_blank",
                ),
                width=2,
                style=page_banner_image_style,
            ),
        ],
        style={"margin-top": "10px"},
    )
    return html.Div(
        children=[
            html.Div(
                banner_content,
                style=page_banner_body,
            ),
            html.Div(
                style=page_banner_border,
            ),
        ],
        style={"width": "100%"},
    )
