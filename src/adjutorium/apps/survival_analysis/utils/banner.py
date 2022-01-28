# third party
from dash import html
import dash_bootstrap_components as dbc

# adjutorium absolute
from adjutorium.apps.survival_analysis.assets.styles import (
    page_banner_body,
    page_banner_border,
    page_banner_button_right_style,
    page_banner_button_style,
    page_banner_image_style,
    page_banner_link_title_style,
    page_banner_title_style,
)


def banner_template(title: str) -> html.Div:
    banner_content = dbc.Row(
        [
            html.H2(
                children=[
                    html.A(
                        title,
                        href="/",
                        style=page_banner_link_title_style,
                    )
                ],
                style=page_banner_title_style,
            ),
            html.A(
                dbc.Button("Send Feedback"),
                href="https://www.vanderschaar-lab.com/contact-us/",
                target="_blank",
                style=page_banner_button_style,
            ),
            html.A(
                dbc.Button("Learn more"),
                href="https://www.vanderschaar-lab.com/",
                target="_blank",
                style=page_banner_button_right_style,
            ),
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
                style=page_banner_image_style,
            ),
        ]
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
