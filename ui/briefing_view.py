"""Partner briefing screen.

Look up an investor, get a briefing built from public sources and cached in
Supabase. Research and generation live in ``services.briefing_service``.
"""

from __future__ import annotations

import streamlit as st

from services.briefing_service import (
    DEFAULT_MAX_AGE_HOURS,
    BriefingConfigurationError,
    BriefingError,
    BriefingService,
    PartnerBriefing,
)

_BRIEFING_KEY = "partner_briefing"


@st.cache_resource(show_spinner=False)
def _service() -> BriefingService:
    return BriefingService()


# --------------------------------------------------------------------------- #
# Form
# --------------------------------------------------------------------------- #
def _render_form() -> dict | None:
    with st.form("briefing_form"):
        col1, col2 = st.columns(2)
        with col1:
            partner_name = st.text_input(
                "Investor name", placeholder="e.g. Jane Doe"
            )
        with col2:
            firm = st.text_input(
                "Firm", placeholder="e.g. Blume Ventures",
                help="Optional, but it sharpens the search a lot.",
            )

        context = st.text_input(
            "What are you raising? (optional)",
            placeholder="e.g. $2M seed for a vertical AI copilot for CFOs",
            help="Used to tailor the opener and the questions to expect.",
        )

        col3, col4 = st.columns([1, 1])
        with col3:
            max_age_days = st.slider(
                "Refresh briefings older than (days)", 1, 30,
                int(DEFAULT_MAX_AGE_HOURS // 24),
            )
        with col4:
            force_refresh = st.checkbox(
                "Force fresh research",
                help="Skip the cache and search the web again.",
            )

        submitted = st.form_submit_button("Build briefing", type="primary")

    if not submitted:
        return None
    if not partner_name.strip():
        st.warning("Enter the investor's name.")
        return None

    return {
        "partner_name": partner_name.strip(),
        "firm": firm.strip(),
        "context": context.strip(),
        "max_age_hours": max_age_days * 24,
        "force_refresh": force_refresh,
    }


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def _bullets(title: str, items: list[str]) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    for item in items:
        st.markdown(f"- {item}")


def _render_briefing(briefing: PartnerBriefing) -> None:
    st.divider()

    heading = briefing.partner_name + (f" — {briefing.firm}" if briefing.firm else "")
    st.subheader(heading)

    origin = "from cache" if briefing.from_cache else "freshly researched"
    stamp = briefing.generated_at[:16].replace("T", " ") if briefing.generated_at else ""
    st.caption(f"{origin}{f' · {stamp} UTC' if stamp else ''}")

    if briefing.summary:
        st.write(briefing.summary)

    left, right = st.columns(2, gap="large")
    with left:
        _bullets("Invests in", briefing.investment_focus)
        _bullets("Notable investments", briefing.notable_investments)
    with right:
        _bullets("What they look for", briefing.what_they_look_for)
        _bullets("Questions to expect", briefing.likely_questions)

    if briefing.how_to_open:
        st.markdown("**How to open**")
        with st.container(border=True):
            st.write(briefing.how_to_open)

    if briefing.unknowns:
        with st.expander("Not established by the sources — verify these"):
            for item in briefing.unknowns:
                st.markdown(f"- {item}")

    if briefing.sources:
        with st.expander(f"Sources ({len(briefing.sources)})"):
            for source in briefing.sources:
                st.markdown(f"- [{source.title}]({source.url})")

    st.caption(
        "Built from public web sources. Verify anything you plan to say in the "
        "room — search results go stale and can be wrong."
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render() -> None:
    """Render the partner briefing screen."""
    st.title("Partner briefing")
    st.caption(
        "Research an investor before the meeting. Briefings are cached in "
        "Supabase, so looking up the same partner again is instant."
    )

    params = _render_form()

    if params is not None:
        try:
            with st.spinner(f"Researching {params['partner_name']}…"):
                briefing = _service().get_briefing(**params)
        except BriefingConfigurationError as exc:
            st.error(f"Configuration problem: {exc}")
            return
        except ValueError as exc:
            st.warning(str(exc))
            return
        except BriefingError as exc:
            st.error(f"Could not build the briefing. {exc}")
            return

        st.session_state[_BRIEFING_KEY] = briefing
        if briefing.from_cache:
            st.info("Served from the cache. Tick 'Force fresh research' to rebuild.")
        else:
            st.success(f"Briefing built from {len(briefing.sources)} source(s).")

    briefing = st.session_state.get(_BRIEFING_KEY)
    if briefing is not None:
        _render_briefing(briefing)
