from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[1] / "app.py"


def test_dashboard_renders_summary_without_errors() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "$1,598 per month"
    captions = [caption.value for caption in app.caption]
    # Escaped dollar signs stop Streamlit from rendering the text between them as LaTeX.
    assert (
        "Text summary: Average rent was \\$603 in 2000 and \\$1,598 in 2025."
        in captions
    )
