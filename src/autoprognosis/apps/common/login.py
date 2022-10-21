# third party
import streamlit as st


def is_authenticated(dummy):
    return dummy == "autoprognosis"


def generate_login_block():
    block1 = st.empty()
    block2 = st.empty()

    return block1, block2


def clean_blocks(blocks):
    for block in blocks:
        block.empty()


def login(blocks):
    blocks[0].markdown(
        """
            <style>
                input {
                    -webkit-text-security: disc;
                }
            </style>
        """,
        unsafe_allow_html=True,
    )

    return blocks[1].text_input("Password")
