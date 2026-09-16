"""Pitch editor screen.

Founder pastes a pitch script; the critique and the rewrite are shown
side-by-side. No API calls live in this module.
"""

from __future__ import annotations

import json

import streamlit as st

from services.investor_matcher import SECTORS, STAGES
from services.pitch_editor import (
    MAX_PITCH_CHARS,
    PITCH_FORMATS,
    CritiquePoint,
    PitchEditor,
    PitchEditorConfigurationError,
    PitchEditorError,
    PitchReview,
)

_REVIEW_KEY = "pitch_review"

_SEVERITY_BADGE = {
    "high": "🔴 High",
    "medium": "🟠 Medium",
    "low": "🟡 Polish",
}


@st.cache_resource(show_spinner=False)
def _editor() -> PitchEditor:
    return PitchEditor()


# --------------------------------------------------------------------------- #
# Form
# --------------------------------------------------------------------------- #
def _render_form() -> dict | None:
    """Draw the input form. Returns review kwargs once submitted."""
    with st.form("pitch_editor_form"):
        pitch_text = st.text_area(
            "Paste your pitch script",
            height=260,
            max_chars=MAX_PITCH_CHARS,
            placeholder=(
                "Paste the script you would actually say out loud — not bullet "
                "points from a deck."
            ),
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            company = st.text_input("Company (optional)")
        with col2:
            sector = st.selectbox("Sector (optional)", [""] + SECTORS, index=0)
        with col3:
            stage = st.selectbox("Stage (optional)", [""] + STAGES, index=0)

        col4, col5 = st.columns([2, 1])
        with col4:
            pitch_format = st.selectbox("Rewrite as", PITCH_FORMATS, index=1)
        with col5:
            target_words = st.number_input(
                "Target words", min_value=0, max_value=1200, value=0, step=25,
                help="0 lets the partner pick a length for the chosen format.",
            )

        submitted = st.form_submit_button("Run the partner review", type="primary")

    if not submitted:
        return None

    return {
        "pitch_text": pitch_text,
        "company": company,
        "sector": sector,
        "stage": stage,
        "pitch_format": pitch_format,
        "target_words": int(target_words) or None,
    }


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def _render_point(index: int, point: CritiquePoint) -> None:
    with st.container(border=True):
        st.markdown(
            f"**{index}. {point.flaw}**  \n"
            f"{_SEVERITY_BADGE.get(point.severity, point.severity)}"
        )
        if point.why_it_matters:
            st.caption(point.why_it_matters)
        if point.fix:
            st.markdown(f"→ *{point.fix}*")


def _render_results() -> None:
    review: PitchReview | None = st.session_state.get(_REVIEW_KEY)
    if review is None:
        return

    st.divider()
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("What a partner would push back on")
        high = sum(1 for p in review.critique_points if p.severity == "high")
        st.caption(
            f"{len(review.critique_points)} point(s)"
            + (f" · {high} likely to lose the room" if high else "")
        )
        for index, point in enumerate(review.critique_points, start=1):
            _render_point(index, point)

    with right:
        st.subheader("Rewritten pitch")
        st.caption(
            f"{review.original_word_count} → {review.rewritten_word_count} words "
            "· square brackets mark numbers you still need to supply"
        )
        with st.container(border=True):
            st.write(review.rewritten_pitch)

        st.download_button(
            "Download rewrite (.txt)",
            data=review.rewritten_pitch.encode("utf-8"),
            file_name="rewritten_pitch.txt",
            mime="text/plain",
        )

    with st.expander("Compare with your original"):
        orig_col, new_col = st.columns(2, gap="large")
        with orig_col:
            st.markdown("**Original**")
            st.text(review.original_pitch)
        with new_col:
            st.markdown("**Rewritten**")
            st.text(review.rewritten_pitch)

    with st.expander("Raw JSON"):
        st.code(json.dumps(review.as_dict(), indent=2, ensure_ascii=False), language="json")

    st.caption(
        "The reviewer persona is a simulation for practice. It is not affiliated "
        "with, or endorsed by, any investment firm."
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render() -> None:
    """Render the pitch editor screen."""
    st.title("Pitch editor")
    st.caption(
        "Paste a pitch script and get the critique a blunt partner would give "
        "you, plus a rewritten version."
    )

    params = _render_form()

    if params is not None:
        try:
            with st.spinner("Reading your pitch the way a partner would…"):
                review = _editor().review(**params)
        except PitchEditorConfigurationError as exc:
            st.error(f"Configuration problem: {exc}")
            return
        except ValueError as exc:
            st.warning(str(exc))
            return
        except PitchEditorError as exc:
            st.error(f"Could not review the pitch. {exc}")
            return

        st.session_state[_REVIEW_KEY] = review
        st.success(
            f"{len(review.critique_points)} critique point(s) and a rewrite ready."
        )

    _render_results()
