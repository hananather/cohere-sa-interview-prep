"""Streamlit Community Cloud entrypoint for the Defence Agent backup demo.

This app intentionally serves the deterministic presentation backup instead of
running live Cohere calls. Use `streamlit_app.py` locally for the full app.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
BACKUP_PAGE = ROOT / "presentation_backup" / "index.html"


def main() -> None:
    st.set_page_config(
        page_title="Defence Agent Demo",
        page_icon="DA",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {
            max-width: 1240px;
            padding: 0.5rem 1rem 2rem;
        }
        iframe {
            border: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if not BACKUP_PAGE.exists():
        st.error("Backup page is missing. Run `python presentation_backup/build_backup.py`.")
        return
    html = BACKUP_PAGE.read_text(encoding="utf-8")
    st.html(html)


if __name__ == "__main__":
    main()
