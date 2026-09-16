"""Match a founder's startup profile against the investor list.

Business logic and the Gemini API call live here; ``ui/`` only collects the
form input and renders whatever this module returns.

Environment variables
---------------------
GEMINI_API_KEY      (required)
GEMINI_MODEL        (optional) defaults to ``DEFAULT_MODEL`` below.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from services import llm
from services.llm import (
    DEFAULT_MODEL,
    MAX_RETRIES,
    Client,
    LLMConfigurationError,
    LLMError,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 5

# Sending 800 investors to the model is slow, expensive and mostly noise, so a
# cheap heuristic shortlists candidates before the model ranks them.
MAX_CANDIDATES = 60

SECTORS = [
    "AI / ML", "SaaS", "Fintech", "Consumer Internet", "D2C / Commerce",
    "Healthtech", "Edtech", "Climate / Energy", "Deeptech", "Space",
    "Logistics / Supply Chain", "Agritech", "Gaming", "Cybersecurity",
    "Devtools / Infrastructure", "Marketplaces", "Robotics", "Biotech",
    "Web3 / Crypto", "Other",
]

STAGES = [
    "Pre-seed", "Seed", "Seed+ / Bridge", "Series A", "Series B", "Series C+",
]

# Investor rows come from Supabase and column naming varies between imports,
# so each logical field is looked up under several plausible names.
_FIELD_ALIASES: Dict[str, Sequence[str]] = {
    "name": ("name", "investor_name", "full_name", "partner_name", "contact"),
    "firm": ("firm", "fund", "fund_name", "organisation", "organization", "company"),
    "sectors": ("sectors", "sector", "focus", "focus_areas", "verticals", "themes"),
    "stages": ("stages", "stage", "stage_focus", "investment_stage"),
    "check_size": ("check_size", "cheque_size", "ticket_size", "typical_check", "check"),
    "geography": ("geography", "geographies", "location", "region", "markets", "country"),
    "thesis": ("thesis", "notes", "description", "bio", "about", "summary"),
    "website": ("website", "url", "site", "link", "domain"),
}


class MatcherError(LLMError):
    """Raised when a match cannot be produced."""


class MatcherConfigurationError(MatcherError, LLMConfigurationError):
    """The Gemini API key or model configuration is missing/invalid."""


# --------------------------------------------------------------------------- #
# Data objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StartupProfile:
    """What the founder tells us about the round they are raising."""

    sector: str
    stage: str
    raise_amount: float
    currency: str = "USD"          # "USD" or "INR"
    description: str = ""
    geography: str = ""
    traction: str = ""

    def __post_init__(self) -> None:
        if not str(self.sector).strip():
            raise ValueError("sector is required.")
        if not str(self.stage).strip():
            raise ValueError("stage is required.")
        if self.raise_amount is None or float(self.raise_amount) <= 0:
            raise ValueError("raise_amount must be greater than zero.")
        if self.currency not in ("USD", "INR"):
            raise ValueError("currency must be 'USD' or 'INR'.")

    @property
    def raise_label(self) -> str:
        """Human-readable round size — $ in millions, ₹ in crore."""
        if self.currency == "INR":
            return f"₹{self.raise_amount:,.2f} crore"
        return f"${self.raise_amount:,.2f}M"

    def as_prompt_block(self) -> str:
        lines = [
            f"Sector: {self.sector}",
            f"Stage: {self.stage}",
            f"Raise amount: {self.raise_label}",
        ]
        if self.geography:
            lines.append(f"Geography: {self.geography}")
        if self.traction:
            lines.append(f"Traction: {self.traction}")
        if self.description:
            lines.append(f"What they do: {self.description}")
        return "\n".join(lines)


@dataclass
class InvestorMatch:
    """One ranked investor, with the model's reasoning attached."""

    investor: Dict[str, Any]
    score: int
    explanation: str
    concerns: str = ""
    rank: int = 0

    @property
    def name(self) -> str:
        return _field(self.investor, "name") or _field(self.investor, "firm") or "Unknown"

    @property
    def firm(self) -> str:
        return _field(self.investor, "firm")

    @property
    def website(self) -> str:
        return _field(self.investor, "website")

    def as_row(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "name": self.name,
            "firm": self.firm,
            "score": self.score,
            "explanation": self.explanation,
            "concerns": self.concerns,
            "website": self.website,
        }


# --------------------------------------------------------------------------- #
# Field helpers
# --------------------------------------------------------------------------- #
def _field(row: Dict[str, Any], logical_name: str) -> str:
    """Read a logical field from an investor row whatever it is called."""
    for key in _FIELD_ALIASES.get(logical_name, (logical_name,)):
        if key in row and row[key] not in (None, "", [], {}):
            value = row[key]
            if isinstance(value, (list, tuple, set)):
                return ", ".join(str(v) for v in value if v)
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False)
            return str(value).strip()
    return ""


def _row_id(row: Dict[str, Any], fallback_index: int) -> str:
    for key in ("id", "uuid", "investor_id", "slug"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return f"idx-{fallback_index}"


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9+]+", text.lower()) if len(t) > 2}


_truncate = llm.truncate


def _two_sentences(text: str) -> str:
    """Trim an explanation to at most two sentences."""
    return llm.limit_sentences(text, 2)


# --------------------------------------------------------------------------- #
# Tool schema — forces well-formed JSON back from the model
# --------------------------------------------------------------------------- #
def _match_tool(top_n: int) -> Dict[str, Any]:
    return {
        "name": "submit_matches",
        "description": (
            "The ranked investor matches for this startup, best first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "matches": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": top_n,
                    "description": (
                        f"The {top_n} best-fitting investors, best first. "
                        "Only use investors from the supplied list."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "investor_id": {
                                "type": "string",
                                "description": "The id exactly as given in the list.",
                            },
                            "investor_name": {
                                "type": "string",
                                "description": "The investor's name, for verification.",
                            },
                            "score": {
                                "type": "integer",
                                "description": "Fit from 0-100. Be discriminating; "
                                               "reserve 85+ for genuinely strong fits.",
                            },
                            "explanation": {
                                "type": "string",
                                "description": (
                                    "Exactly two sentences on why this investor fits: "
                                    "the first on sector/stage/cheque-size alignment, "
                                    "the second on the concrete angle the founder should "
                                    "lead with. Cite only facts present in the investor "
                                    "record."
                                ),
                            },
                            "concerns": {
                                "type": "string",
                                "description": "Optional short caveat, or an empty string.",
                            },
                        },
                        "required": ["investor_id", "investor_name", "score", "explanation"],
                    },
                }
            },
            "required": ["matches"],
        },
    }


SYSTEM_PROMPT = (
    "You are an analyst on a venture capital platform team. You match founders "
    "to investors using only the investor records supplied to you.\n\n"
    "Rules:\n"
    "- Rank on real signal: sector focus, stage focus, cheque size versus the "
    "round being raised, and geography.\n"
    "- Never invent an investor, a fund, a thesis, or a cheque size. If a record "
    "is thin, say so in the concerns field and score it lower.\n"
    "- Prefer a precise, unglamorous fit over a famous name with a weak fit.\n"
    "- Every explanation must be exactly two sentences, specific to this startup, "
    "and free of filler like 'great fit' or 'perfect match'.\n"
    "- Return your answer as JSON matching the required schema, nothing else."
)


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class InvestorMatcher:
    """Ranks investors against a startup profile using the Gemini API."""

    def __init__(
        self,
        client: Optional[Client] = None,
        model: str = DEFAULT_MODEL,
        max_candidates: int = MAX_CANDIDATES,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.model = model
        self.max_candidates = max_candidates
        self.max_retries = max_retries
        self._client = client or self._build_client()

    @staticmethod
    def _build_client() -> Client:
        try:
            return llm.build_client()
        except LLMConfigurationError as exc:
            raise MatcherConfigurationError(str(exc)) from exc

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def match(
        self,
        profile: StartupProfile,
        investors: Sequence[Dict[str, Any]],
        top_n: int = DEFAULT_TOP_N,
    ) -> List[InvestorMatch]:
        """Return the top ``top_n`` investors for ``profile``, best first.

        Args:
            profile: The founder's startup profile.
            investors: Investor rows as returned by ``DBClient.fetch_all_investors``.
            top_n: How many matches to return.

        Returns:
            Ranked ``InvestorMatch`` objects. Shorter than ``top_n`` if the
            investor list is small or the model returned fewer usable matches.

        Raises:
            ValueError: the investor list is empty or ``top_n`` < 1.
            MatcherError: the API call failed or returned nothing usable.
        """
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        rows = [r for r in investors if isinstance(r, dict) and r]
        if not rows:
            raise ValueError(
                "No investors to match against. Check that the investors table "
                "has rows and that the current user can read it."
            )

        indexed = {_row_id(row, i): row for i, row in enumerate(rows)}
        shortlist = self._shortlist(profile, indexed)
        logger.info(
            "Matching %s / %s against %d of %d investors",
            profile.sector, profile.stage, len(shortlist), len(indexed),
        )

        payload = self._call_model(profile, shortlist, top_n)
        matches = self._build_matches(payload, indexed, top_n)
        if not matches:
            raise MatcherError(
                "The model did not return any investor from the supplied list. "
                "Try again, or widen the sector/stage filters."
            )
        return matches

    # ------------------------------------------------------------------ #
    # Shortlist
    # ------------------------------------------------------------------ #
    def _shortlist(
        self, profile: StartupProfile, indexed: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Cheap keyword pre-filter. Never drops rows below the cap."""
        if len(indexed) <= self.max_candidates:
            return indexed

        sector_tokens = _tokens(profile.sector) | _tokens(profile.description)
        stage_tokens = _tokens(profile.stage)

        def cheap_score(row: Dict[str, Any]) -> int:
            sectors = _tokens(_field(row, "sectors") + " " + _field(row, "thesis"))
            stages = _tokens(_field(row, "stages"))
            score = 3 * len(sector_tokens & sectors) + 4 * len(stage_tokens & stages)
            if profile.geography and _tokens(profile.geography) & _tokens(
                _field(row, "geography")
            ):
                score += 2
            # A record with nothing filled in cannot be judged, so deprioritise it.
            if not sectors and not stages:
                score -= 1
            return score

        ranked = sorted(indexed.items(), key=lambda kv: cheap_score(kv[1]), reverse=True)
        return dict(ranked[: self.max_candidates])

    # ------------------------------------------------------------------ #
    # Model call
    # ------------------------------------------------------------------ #
    @staticmethod
    def _render_investors(indexed: Dict[str, Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for inv_id, row in indexed.items():
            parts = [f"<investor id=\"{inv_id}\">"]
            for label, logical in (
                ("name", "name"), ("firm", "firm"), ("sectors", "sectors"),
                ("stages", "stages"), ("check_size", "check_size"),
                ("geography", "geography"), ("thesis", "thesis"),
            ):
                value = _field(row, logical)
                if value:
                    parts.append(f"  {label}: {_truncate(value, 400)}")
            parts.append("</investor>")
            blocks.append("\n".join(parts))
        return "\n".join(blocks)

    def _call_model(
        self,
        profile: StartupProfile,
        indexed: Dict[str, Dict[str, Any]],
        top_n: int,
    ) -> Dict[str, Any]:
        tool = _match_tool(top_n)
        user_message = (
            "<startup>\n"
            f"{profile.as_prompt_block()}\n"
            "</startup>\n\n"
            "<investors>\n"
            f"{self._render_investors(indexed)}\n"
            "</investors>\n\n"
            f"Pick the {top_n} investors from the list above that best fit this "
            "startup and return them as JSON, best first."
        )

        try:
            return llm.call_tool(
                client=self._client,
                model=self.model,
                system=SYSTEM_PROMPT,
                user_message=user_message,
                tool=tool,
                max_tokens=2048,
                max_retries=self.max_retries,
            )
        except LLMConfigurationError as exc:
            raise MatcherConfigurationError(str(exc)) from exc
        except MatcherError:
            raise
        except LLMError as exc:
            raise MatcherError(str(exc)) from exc

    # ------------------------------------------------------------------ #
    # Result assembly
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_matches(
        payload: Dict[str, Any],
        indexed: Dict[str, Dict[str, Any]],
        top_n: int,
    ) -> List[InvestorMatch]:
        """Map the model's output back onto real rows, dropping anything invented."""
        raw = payload.get("matches")
        if not isinstance(raw, list):
            raise MatcherError("The model's response did not contain a match list.")

        # Fallback lookup so a correct name with a mangled id still resolves.
        by_name = {
            _field(row, "name").casefold(): key
            for key, row in indexed.items()
            if _field(row, "name")
        }

        matches: List[InvestorMatch] = []
        seen: set[str] = set()

        for item in raw:
            if not isinstance(item, dict):
                continue
            key = str(item.get("investor_id", "")).strip()
            if key not in indexed:
                key = by_name.get(str(item.get("investor_name", "")).strip().casefold(), "")
            if not key or key in seen:
                if key:
                    logger.debug("Skipping duplicate match %s", key)
                else:
                    logger.warning(
                        "Model returned an unknown investor: %r", item.get("investor_name")
                    )
                continue

            try:
                score = int(item.get("score", 0))
            except (TypeError, ValueError):
                score = 0

            explanation = _two_sentences(item.get("explanation", ""))
            if not explanation:
                explanation = "No explanation was returned for this match."

            seen.add(key)
            matches.append(
                InvestorMatch(
                    investor=indexed[key],
                    score=max(0, min(100, score)),
                    explanation=explanation,
                    concerns=_truncate(item.get("concerns") or "", 300),
                )
            )
            if len(matches) == top_n:
                break

        for position, match in enumerate(matches, start=1):
            match.rank = position
        return matches


__all__ = [
    "InvestorMatcher",
    "StartupProfile",
    "InvestorMatch",
    "MatcherError",
    "MatcherConfigurationError",
    "SECTORS",
    "STAGES",
    "DEFAULT_MODEL",
    "DEFAULT_TOP_N",
]
