"""Entry point for local runs, Hugging Face Spaces, and Render.

Kept at the application root because those platforms launch `python app.py`.
All behaviour lives in the `twin` package.
"""
import gradio as gr

from twin.config import server_name, server_port
from twin.ui import build_ui
from twin.ui.styles import CSS, JS


def main():
    build_ui().launch(
        server_name=server_name(), server_port=server_port(),
        css=CSS, js=JS, theme=gr.themes.Base(),
    )


if __name__ == '__main__':
    main()
