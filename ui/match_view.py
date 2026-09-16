"""Investor-matching screen.

Collects the founder's profile in a Streamlit form, hands it to
``services.investor_matcher``, and renders the ranked result. No API calls and
no SQL live in this module.
"""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List

import streamlit as st

from core.database import DatabaseError, DBClient
from services.investor_matcher import (
    DEFAULT_TOP_N,
    InvestorMatch,
    InvestorMatcher,
    MatcherConfigurationError,
    MatcherError,
    SECTORS,
    STAGES,
    StartupProfile,
)

_RESULT_KEY = "match_results"
_PROFILE_KEY = "match_profile"


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _db_client() -> DBClient:
    return DBClient()


@st.cache_resource(show_spinner=False)
def _matcher() -> InvestorMatcher:
    return InvestorMatcher()


@st.cache_data(ttl=600, show_spinner=False)
def _load_investors() -> List[Dict[str, Any]]:
    """Investor list, cached for 10 minutes so repeat matches don't re-query."""
    return _db_client().fetch_all_investors()


# --------------------------------------------------------------------------- #
# Form
# --------------------------------------------------------------------------- #
def _render_form() -> StartupProfile | None:
    """Draw the input form. Returns a profile once submitted and valid."""
    with st.form("investor_match_form", clear_on_submit=False):
        st.subheader("Startup profile")

        col1, col2 = st.columns(2)
        with col1:
            sector = st.selectbox("Sector", SECTORS, index=0)
            custom_sector = st.text_input(
                "Specify sector",
                placeholder="e.g. vertical AI for insurance",
                help="Only used when Sector is set to 'Other'.",
            )
        with col2:
            stage = st.selectbox("Stage", STAGES, index=STAGES.index("Seed"))
            geography = st.text_input(
                "Geography", placeholder="e.g. India, SEA, US"
            )

        col3, col4 = st.columns([1, 2])
        with col3:
            currency = st.radio(
                "Currency", ["USD", "INR"], horizontal=True,
                help="USD amounts are in millions, INR amounts in crore.",
            )
        with col4:
            raise_amount = st.number_input(
                "Raise amount",
                min_value=0.0, value=1.5, step=0.25, format="%.2f",
                help="Millions for USD, crore for INR.",
            )

        description = st.text_area(
            "What does the startup do?",
            placeholder="One or two lines — the sharper this is, the better the matches.",
            height=80,
        )
        traction = st.text_input(
            "Traction (optional)", placeholder="e.g. $40k MRR, growing 18% MoM"
        )

        top_n = st.slider("Number of matches", 3, 10, DEFAULT_TOP_N)

        submitted = st.form_submit_button("Find investors", type="primary")

    if not submitted:
        return None

    resolved_sector = custom_sector.strip() if sector == "Other" else sector
    if sector == "Other" and not resolved_sector:
        st.warning("Please specify the sector, or pick one from the list.")
        return None

    try:
        profile = StartupProfile(
            sector=resolved_sector,
            stage=stage,
            raise_amount=float(raise_amount),
            currency=currency,
            description=description.strip(),
            geography=geography.strip(),
            traction=traction.strip(),
        )
    except ValueError as exc:
        st.warning(str(exc))
        return None

    st.session_state["_top_n"] = top_n
    return profile


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def _score_tone(score: int) -> str:
    if score >= 80:
        return "🟢"
    if score >= 60:
        return "🟡"
    return "⚪"


def _render_match(match: InvestorMatch) -> None:
    with st.container(border=True):
        head, meter = st.columns([4, 1])
        with head:
            title = match.name
            if match.firm and match.firm != match.name:
                title = f"{match.name} — {match.firm}"
            st.markdown(f"**{match.rank}. {title}**")
            if match.website:
                st.caption(match.website)
        with meter:
            st.metric("Fit", f"{_score_tone(match.score)} {match.score}")

        st.write(match.explanation)
        if match.concerns:
            st.caption(f"⚠️ {match.concerns}")

        with st.expander("Investor record"):
            st.json(match.investor, expanded=False)


def _results_csv(matches: List[InvestorMatch]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["rank", "name", "firm", "score", "explanation", "concerns", "website"],
    )
    writer.writeheader()
    for match in matches:
        writer.writerow(match.as_row())
    return buffer.getvalue().encode("utf-8")


def _render_results() -> None:
    matches: List[InvestorMatch] = st.session_state.get(_RESULT_KEY, [])
    profile: StartupProfile | None = st.session_state.get(_PROFILE_KEY)
    if not matches or profile is None:
        return

    st.divider()
    st.subheader(f"Top {len(matches)} matches")
    st.caption(
        f"{profile.sector} · {profile.stage} · raising {profile.raise_label}"
        + (f" · {profile.geography}" if profile.geography else "")
    )

    for match in matches:
        _render_match(match)

    st.download_button(
        "Download as CSV",
        data=_results_csv(matches),
        file_name="investor_matches.csv",
        mime="text/csv",
    )
    st.caption(
        "Matches are generated from the investor records in Supabase. "
        "Verify a fund's current mandate before reaching out."
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render() -> None:
    """Render the investor-matching screen."""
    st.title("Investor matching")
    st.caption("Describe the round, and Moneypenny ranks the investor list against it.")

    profile = _render_form()

    if profile is not None:
        try:
            with st.spinner("Loading investors…"):
                investors = _load_investors()
        except DatabaseError as exc:
            st.error(f"Could not load investors. {exc}")
            return

        if not investors:
            st.warning(
                "The investors table is empty, so there is nothing to match against."
            )
            return

        try:
            with st.spinner(f"Evaluating {len(investors)} investors…"):
                matches = _matcher().match(
                    profile, investors, top_n=st.session_state.get("_top_n", DEFAULT_TOP_N)
                )
        except MatcherConfigurationError as exc:
            st.error(f"Configuration problem: {exc}")
            return
        except (MatcherError, ValueError) as exc:
            st.error(f"Could not generate matches. {exc}")
            return

        st.session_state[_RESULT_KEY] = matches
        st.session_state[_PROFILE_KEY] = profile
        st.success(f"Found {len(matches)} matches out of {len(investors)} investors.")

    _render_results()
