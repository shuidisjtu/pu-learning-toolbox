"""Console launcher for the optional Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    """Launch the packaged Streamlit application."""
    try:
        from streamlit.web import cli as streamlit_cli
    except ImportError as exc:
        raise SystemExit(
            'The UI requires optional dependencies. Install them with: pip install "pu-toolbox[ui]"'
        ) from exc

    app_path = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app_path), "--browser.gatherUsageStats=false"]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
