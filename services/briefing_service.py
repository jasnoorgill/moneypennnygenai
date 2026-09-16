"""Partner briefings: research an investor, write a briefing, cache it.

Flow: check the Supabase cache → if the cached briefing is missing or stale,
search the web with Tavily → have Gemini write the briefing from those
results only → write it back to the cache.

Environment variables
---------------------
GEMINI_API_KEY      (required)
TAVILY_API_KEY      (required for fresh research; cached briefings still read)
GEMINI_MODEL        (optional)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from core import config
from core.database import DatabaseError, DBClient
from services import llm
from services.llm import (
    DEFAULT_MODEL,
    MAX_RETRIES,
    Client,
    LLMConfigurationError,
    LLMError,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_AGE_HOURS = 24 * 7      # a week-old briefing is still useful
SEARCH_RESULTS_PER_QUERY = 5
MAX_SOURCE_CHARS = 2_500
MAX_BRIEFING_TOKENS = 3000


class BriefingError(LLMError):
    """A briefing could not be produced."""


class BriefingConfigurationError(BriefingError, LLMConfigurationError):
    """An API key or model configuration is missing/invalid."""


class ResearchError(BriefingError):
    """Web research failed."""


# --------------------------------------------------------------------------- #
# Data objects
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    """One web result the briefing was written from."""

    title: str
    url: str
    snippet: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


@dataclass
class PartnerBriefing:
    """What to know before walking into a meeting with this investor."""

    partner_name: str
    firm: str = ""
    summary: str = ""
    investment_focus: List[str] = field(default_factory=list)
    notable_investments: List[str] = field(default_factory=list)
    what_they_look_for: List[str] = field(default_factory=list)
    likely_questions: List[str] = field(default_factory=list)
    how_to_open: str = ""
    unknowns: List[str] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    from_cache: bool = False
    generated_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """The shape stored in the ``briefing`` column."""
        return {
            "partner_name": self.partner_name,
            "firm": self.firm,
            "summary": self.summary,
            "investment_focus": self.investment_focus,
            "notable_investments": self.notable_investments,
            "what_they_look_for": self.what_they_look_for,
            "likely_questions": self.likely_questions,
            "how_to_open": self.how_to_open,
            "unknowns": self.unknowns,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], sources: Optional[Sequence[Any]] = None
    ) -> "PartnerBriefing":
        def as_list(value: Any) -> List[str]:
            if isinstance(value, str):
                return [value] if value.strip() else []
            if isinstance(value, (list, tuple)):
                return [str(v).strip() for v in value if str(v).strip()]
            return []

        parsed_sources: List[Source] = []
        for item in sources or []:
            if isinstance(item, Source):
                parsed_sources.append(item)
            elif isinstance(item, dict) and item.get("url"):
                parsed_sources.append(
                    Source(
                        title=str(item.get("title") or item["url"]),
                        url=str(item["url"]),
                        snippet=str(item.get("snippet") or ""),
                    )
                )

        return cls(
            partner_name=str(data.get("partner_name") or ""),
            firm=str(data.get("firm") or ""),
            summary=str(data.get("summary") or ""),
            investment_focus=as_list(data.get("investment_focus")),
            notable_investments=as_list(data.get("notable_investments")),
            what_they_look_for=as_list(data.get("what_they_look_for")),
            likely_questions=as_list(data.get("likely_questions")),
            how_to_open=str(data.get("how_to_open") or ""),
            unknowns=as_list(data.get("unknowns")),
            sources=parsed_sources,
        )


# --------------------------------------------------------------------------- #
# Research
# --------------------------------------------------------------------------- #
def search_partner(
    partner_name: str,
    firm: str = "",
    client: Any = None,
    results_per_query: int = SEARCH_RESULTS_PER_QUERY,
) -> List[Source]:
    """Search the web for public information about an investor.

    Args:
        partner_name: The investor's name.
        firm: Their firm, which sharpens the query considerably.
        client: A Tavily client. Built from ``TAVILY_API_KEY`` when omitted.
        results_per_query: Results to request per query.

    Returns:
        Deduplicated sources, best first. May be empty.

    Raises:
        BriefingConfigurationError: ``TAVILY_API_KEY`` is missing or rejected.
        ResearchError: every search failed.
    """
    client = client or build_tavily_client()
    subject = f"{partner_name} {firm}".strip()
    queries = [
        f"{subject} venture capital partner investment thesis",
        f"{subject} portfolio investments led rounds",
        f"{subject} interview what they look for in founders",
    ]

    sources: Dict[str, Source] = {}
    failures: List[str] = []

    for query in queries:
        try:
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=results_per_query,
            )
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            if "api key" in lowered or "unauthorized" in lowered or "401" in lowered:
                raise BriefingConfigurationError(
                    f"Tavily rejected the API key: {message}. Check TAVILY_API_KEY."
                ) from exc
            logger.warning("Tavily search failed for %r: %s", query, exc)
            failures.append(message)
            continue

        for result in (response or {}).get("results", []) or []:
            url = str(result.get("url") or "").strip()
            if not url or url in sources:
                continue
            sources[url] = Source(
                title=str(result.get("title") or url),
                url=url,
                snippet=llm.truncate(result.get("content") or "", MAX_SOURCE_CHARS),
            )

    if not sources and failures:
        raise ResearchError(
            f"Web research failed for '{subject}': {failures[0]}"
        )

    logger.info("Found %d source(s) for %s", len(sources), subject)
    return list(sources.values())


def build_tavily_client() -> Any:
    """Construct a Tavily client from ``TAVILY_API_KEY``.

    Raises:
        BriefingConfigurationError: the key or the package is missing.
    """
    api_key = config.tavily_key()
    if not api_key:
        raise BriefingConfigurationError(
            "TAVILY_API_KEY is not set, so fresh partner research is "
            "unavailable. Add it to your environment or .env file."
        )
    try:
        from tavily import TavilyClient
    except ImportError as exc:  # pragma: no cover
        raise BriefingConfigurationError(
            "tavily-python is not installed. Run: pip install tavily-python"
        ) from exc
    try:
        return TavilyClient(api_key=api_key)
    except Exception as exc:
        raise BriefingConfigurationError(
            f"Could not initialise the Tavily client: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You prepare founders for investor meetings. You are writing a briefing on "
    "one investor, using only the search results supplied to you.\n\n"
    "Rules:\n"
    "- Every statement must trace to the supplied sources. If the sources do "
    "not establish something, leave it out and list it under unknowns.\n"
    "- Never guess a thesis, a cheque size, or a portfolio company. An investor "
    "quoted once about marketplaces is not 'a marketplace specialist'.\n"
    "- Prefer the investor's own words — interviews, posts, talks — over "
    "third-party description, and say when a view is dated.\n"
    "- Stick to professional, public information: what they invest in, how they "
    "evaluate, what they have said publicly. Nothing about their personal life.\n"
    "- Be specific and useful to a founder walking into a room in an hour. No "
    "flattery about the investor, no filler.\n"
    "- Write the likely questions as the investor would actually ask them.\n\n"
    "Return your answer as JSON matching the required schema, nothing else."
)

BRIEFING_TOOL: Dict[str, Any] = {
    "name": "submit_briefing",
    "description": "The partner briefing.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Two to four sentences: who they are, where they "
                               "invest, and what matters most about them to a "
                               "founder pitching them.",
            },
            "investment_focus": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Sectors, stages and geographies the sources "
                               "actually support.",
            },
            "notable_investments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Companies the sources tie to this investor. "
                               "Include the round if the sources give it.",
            },
            "what_they_look_for": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Evaluation criteria, drawn from their own "
                               "statements where possible.",
            },
            "likely_questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Questions this investor is likely to ask, phrased "
                               "as they would ask them.",
            },
            "how_to_open": {
                "type": "string",
                "description": "Two or three sentences on the angle to lead with "
                               "and what to avoid.",
            },
            "unknowns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What the sources did not establish, that the "
                               "founder should verify before the meeting.",
            },
        },
        "required": [
            "summary", "investment_focus", "notable_investments",
            "what_they_look_for", "likely_questions", "how_to_open", "unknowns",
        ],
    },
}


def generate_briefing(
    partner_name: str,
    sources: Sequence[Source],
    firm: str = "",
    context: str = "",
    client: Optional[Client] = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = MAX_RETRIES,
) -> PartnerBriefing:
    """Write a briefing from ``sources`` using Gemini.

    Raises:
        ValueError: no partner name, or no sources to write from.
        BriefingConfigurationError: bad key or model.
        BriefingError: the call failed or returned nothing usable.
    """
    partner_name = (partner_name or "").strip()
    if not partner_name:
        raise ValueError("Enter the investor's name.")
    if not sources:
        raise ValueError(
            f"No public sources were found for '{partner_name}'. Try adding "
            "their firm, or check the spelling."
        )

    if client is None:
        client = _build_gemini_client()

    rendered = "\n\n".join(
        f'<source index="{index}" url="{s.url}">\n'
        f"title: {s.title}\n{s.snippet}\n</source>"
        for index, s in enumerate(sources, start=1)
    )
    subject = f"{partner_name}" + (f" ({firm})" if firm else "")
    user_message = (
        f"<investor>{subject}</investor>\n\n"
        + (f"<founder_context>{context}</founder_context>\n\n" if context else "")
        + f"<search_results>\n{rendered}\n</search_results>\n\n"
        "Write the briefing on this investor from the search results above, "
        "and return it as JSON."
    )

    try:
        payload = llm.call_tool(
            client=client,
            model=model,
            system=SYSTEM_PROMPT,
            user_message=user_message,
            tool=BRIEFING_TOOL,
            max_tokens=MAX_BRIEFING_TOKENS,
            max_retries=max_retries,
        )
    except LLMConfigurationError as exc:
        raise BriefingConfigurationError(str(exc)) from exc
    except BriefingError:
        raise
    except LLMError as exc:
        raise BriefingError(str(exc)) from exc

    payload["partner_name"] = partner_name
    payload["firm"] = firm
    briefing = PartnerBriefing.from_dict(payload, sources)
    if not briefing.summary:
        raise BriefingError(
            "The briefing came back without a summary. Try again."
        )
    briefing.generated_at = datetime.now(timezone.utc).isoformat()
    return briefing


def _build_gemini_client() -> Client:
    try:
        return llm.build_client()
    except LLMConfigurationError as exc:
        raise BriefingConfigurationError(str(exc)) from exc


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class BriefingService:
    """Cached partner briefings: read the cache, research on a miss, write back."""

    def __init__(
        self,
        db: Optional[DBClient] = None,
        gemini_client: Optional[Client] = None,
        tavily_client: Any = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self._db = db
        self._gemini = gemini_client
        self._tavily = tavily_client

    @property
    def db(self) -> DBClient:
        if self._db is None:
            self._db = DBClient()
        return self._db

    # ---------------------------------------------------------------- #
    # Cache
    # ---------------------------------------------------------------- #
    def get_cached(
        self, partner_name: str, max_age_hours: Optional[float] = DEFAULT_MAX_AGE_HOURS
    ) -> Optional[PartnerBriefing]:
        """Return a cached briefing, or None on a miss. Never raises on cache misses."""
        try:
            row = self.db.get_partner_briefing(partner_name, max_age_hours=max_age_hours)
        except DatabaseError as exc:
            # A cache that is down should not block a fresh briefing.
            logger.warning("Could not read the briefing cache: %s", exc)
            return None
        if not row:
            return None

        data = row.get("briefing")
        if isinstance(data, str):
            import json

            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Cached briefing for %s is not valid JSON", partner_name)
                return None
        if not isinstance(data, dict):
            return None

        briefing = PartnerBriefing.from_dict(data, row.get("sources") or [])
        briefing.from_cache = True
        briefing.generated_at = str(row.get("updated_at") or row.get("created_at") or "")
        if not briefing.partner_name:
            briefing.partner_name = str(row.get("partner_name") or partner_name)
        return briefing

    def cache(self, briefing: PartnerBriefing) -> bool:
        """Write a briefing to Supabase. Returns False if the write failed."""
        try:
            self.db.insert_partner_briefing(
                partner_name=briefing.partner_name,
                briefing=briefing.as_dict(),
                sources=[s.as_dict() for s in briefing.sources],
                model=self.model,
            )
            return True
        except (DatabaseError, ValueError) as exc:
            # The briefing is still useful even if it could not be stored.
            logger.warning("Could not cache the briefing: %s", exc)
            return False

    # ---------------------------------------------------------------- #
    # Main entry point
    # ---------------------------------------------------------------- #
    def get_briefing(
        self,
        partner_name: str,
        firm: str = "",
        context: str = "",
        max_age_hours: Optional[float] = DEFAULT_MAX_AGE_HOURS,
        force_refresh: bool = False,
    ) -> PartnerBriefing:
        """Return a briefing, from cache when fresh and from research otherwise.

        Args:
            partner_name: The investor to brief on.
            firm: Their firm. Sharpens the search a lot — supply it when known.
            context: Optional note about the founder's round, to tailor the
                opener and likely questions.
            max_age_hours: Cached briefings older than this are refreshed.
            force_refresh: Skip the cache read entirely.

        Returns:
            A ``PartnerBriefing``; ``from_cache`` says where it came from.

        Raises:
            ValueError: no partner name, or no sources found.
            BriefingConfigurationError: a required key is missing or rejected.
            BriefingError: research or generation failed.
        """
        partner_name = (partner_name or "").strip()
        if not partner_name:
            raise ValueError("Enter the investor's name.")

        if not force_refresh:
            cached = self.get_cached(partner_name, max_age_hours)
            if cached:
                logger.info("Cache hit for %s", partner_name)
                return cached

        sources = search_partner(partner_name, firm, client=self._tavily_client())
        briefing = generate_briefing(
            partner_name=partner_name,
            sources=sources,
            firm=firm,
            context=context,
            client=self._gemini_client(),
            model=self.model,
            max_retries=self.max_retries,
        )
        self.cache(briefing)
        return briefing

    # ---------------------------------------------------------------- #
    # Lazy clients — built only when research actually happens
    # ---------------------------------------------------------------- #
    def _tavily_client(self) -> Any:
        if self._tavily is None:
            self._tavily = build_tavily_client()
        return self._tavily

    def _gemini_client(self) -> Client:
        if self._gemini is None:
            self._gemini = _build_gemini_client()
        return self._gemini


__all__ = [
    "BriefingService",
    "PartnerBriefing",
    "Source",
    "search_partner",
    "generate_briefing",
    "build_tavily_client",
    "BriefingError",
    "BriefingConfigurationError",
    "ResearchError",
    "DEFAULT_MAX_AGE_HOURS",
]
