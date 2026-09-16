"""Moneypenny — reusable Streamlit application for VC fundraising.

Entry point: page configuration, sidebar navigation, and dispatch to the views
in ``ui/``. No business logic or database access belongs in this file.
"""

from __future__ import annotations

import logging

import streamlit as st

from core import config
from ui import briefing_view, editor_view, match_view, qa_view

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

PAGES = {
    "Match": ("🎯", match_view.render),
    "Editor": ("✍️", editor_view.render),
    "Briefing": ("🔍", briefing_view.render),
    "Q&A": ("💬", qa_view.render),
}


def _render_sidebar() -> str:
    """Draw the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown("## 💼 Moneypenny")
        st.caption("Fundraising copilot")

        choice = st.radio(
            "Navigation",
            list(PAGES),
            format_func=lambda name: f"{PAGES[name][0]}  {name}",
            label_visibility="collapsed",
        )

        st.divider()
        missing = config.missing_settings()
        if missing:
            st.warning(
                "Not configured: " + ", ".join(missing),
                icon="⚠️",
            )
            st.caption("Set the matching environment variables (or .env) to enable these.")
        else:
            st.caption("✅ All services configured")

    return choice


def main() -> None:
    st.set_page_config(
        page_title="Moneypenny",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    choice = _render_sidebar()
    PAGES[choice][1]()


if __name__ == "__main__":
    main()
