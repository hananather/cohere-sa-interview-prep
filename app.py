"""Streamlit Community Cloud entrypoint for the Defence Agent app.

The deterministic static backup remains available in `presentation_backup/`,
but the hosted demo should run the real Streamlit application.
"""

from __future__ import annotations

from streamlit_app import main


if __name__ == "__main__":
    main()
