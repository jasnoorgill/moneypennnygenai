"""Critique and rewrite a founder's pitch script.

Takes raw pitch text, runs it past a simulated top-tier VC partner, and returns
a structured object with two fields: ``critique_points`` (the flaws) and
``rewritten_pitch`` (a polished version).

Environment variables
---------------------
GEMINI_API_KEY      (required)
GEMINI_MODEL        (optional)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from services import llm
from services.llm import (
    DEFAULT_MODEL,
    MAX_RETRIES,
    Client,
    LLMConfigurationError,
    LLMError,
)

logger = logging.getLogger(__name__)

MIN_PITCH_CHARS = 80
MAX_PITCH_CHARS = 20_000
MAX_OUTPUT_TOKENS = 4096

SEVERITIES = ("high", "medium", "low")
_SEVERITY_ORDER = {level: i for i, level in enumerate(SEVERITIES)}

PITCH_FORMATS = [
    "Elevator pitch (30 seconds)",
    "Investor meeting opener (2 minutes)",
    "Demo day pitch (5 minutes)",
    "Full seed pitch narrative",
    "Cold email to an investor",
]


class PitchEditorError(LLMError):
    """Raised when a review cannot be produced."""


class PitchEditorConfigurationError(PitchEditorError, LLMConfigurationError):
    """The Gemini API key or model configuration is missing/invalid."""


# --------------------------------------------------------------------------- #
# Data objects
# --------------------------------------------------------------------------- #
@dataclass
class CritiquePoint:
    """One flaw the partner would push back on."""

    flaw: str
    why_it_matters: str = ""
    fix: str = ""
    severity: str = "medium"

    def __post_init__(self) -> None:
        self.severity = str(self.severity or "medium").strip().lower()
        if self.severity not in SEVERITIES:
            self.severity = "medium"

    @property
    def sort_key(self) -> int:
        return _SEVERITY_ORDER[self.severity]


@dataclass
class PitchReview:
    """The result of a review: what is wrong, and a rewritten pitch."""

    critique_points: List[CritiquePoint] = field(default_factory=list)
    rewritten_pitch: str = ""
    original_pitch: str = ""
    model: str = ""

    @property
    def original_word_count(self) -> int:
        return len(self.original_pitch.split())

    @property
    def rewritten_word_count(self) -> int:
        return len(self.rewritten_pitch.split())

    def as_dict(self) -> Dict[str, Any]:
        """The plain JSON shape: critique_points + rewritten_pitch."""
        return {
            "critique_points": [asdict(point) for point in self.critique_points],
            "rewritten_pitch": self.rewritten_pitch,
        }


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are role-playing a partner at Andreessen Horowitz (a16z) reviewing a "
    "founder's pitch script before they take it out to market. You are known "
    "for being direct to the point of bluntness: you have sat through thousands "
    "of pitches and you can tell within ninety seconds whether a founder has a "
    "real insight or is performing one.\n\n"
    "How you critique:\n"
    "- Attack substance before style. Vague market claims, an unquantified "
    "problem, no evidence of why now, no wedge, a business model that does not "
    "follow from the product, unfalsifiable traction language, competitors "
    "waved away — these matter far more than word choice.\n"
    "- Name the specific line or claim that fails, and say what a partner would "
    "actually ask in the room.\n"
    "- Be concrete about the fix. 'Add metrics' is useless; 'replace \"strong "
    "early traction\" with the actual MRR and month-on-month growth' is useful.\n"
    "- Do not invent numbers, customers, or facts the founder did not give you. "
    "Where a claim needs evidence the founder has not supplied, say that the "
    "number is missing and mark it as a placeholder in the rewrite, using square "
    "brackets like [X paying customers].\n"
    "- If the pitch is genuinely good in places, say so briefly rather than "
    "manufacturing criticism — but never pad the list to look thorough.\n\n"
    "The rewrite:\n"
    "- Keep the founder's voice and every real fact they gave you. You are "
    "sharpening their pitch, not writing your own.\n"
    "- Open with the insight or the problem, not with the company name and a "
    "list of adjectives. Cut hedging, jargon and filler.\n"
    "- Match the requested format and length.\n"
    "- Return plain prose the founder can read aloud. No headers, no bullet "
    "points, no stage directions.\n\n"
    "Return your answer as JSON matching the required schema, nothing else."
)

REVIEW_TOOL: Dict[str, Any] = {
    "name": "submit_review",
    "description": "The critique and the rewritten pitch.",
    "input_schema": {
        "type": "object",
        "properties": {
            "critique_points": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "description": "The flaws in this pitch, most damaging first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "flaw": {
                            "type": "string",
                            "description": "The flaw in one sentence, naming the "
                                           "specific claim or section at fault.",
                        },
                        "why_it_matters": {
                            "type": "string",
                            "description": "What a partner would think or ask when "
                                           "they hit this. One or two sentences.",
                        },
                        "fix": {
                            "type": "string",
                            "description": "The concrete change to make. One or two "
                                           "sentences.",
                        },
                        "severity": {
                            "type": "string",
                            "enum": list(SEVERITIES),
                            "description": "high = would lose the room; medium = "
                                           "weakens it; low = polish.",
                        },
                    },
                    "required": ["flaw", "why_it_matters", "fix", "severity"],
                },
            },
            "rewritten_pitch": {
                "type": "string",
                "description": "The polished pitch as plain prose, ready to read "
                               "aloud. Preserves the founder's real facts and voice.",
            },
        },
        "required": ["critique_points", "rewritten_pitch"],
    },
}


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
class PitchEditor:
    """Runs a founder's pitch past a simulated VC partner."""

    def __init__(
        self,
        client: Optional[Client] = None,
        model: str = DEFAULT_MODEL,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self._client = client or self._build_client()

    @staticmethod
    def _build_client() -> Client:
        try:
            return llm.build_client()
        except LLMConfigurationError as exc:
            raise PitchEditorConfigurationError(str(exc)) from exc

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def review(
        self,
        pitch_text: str,
        company: str = "",
        sector: str = "",
        stage: str = "",
        pitch_format: str = "Investor meeting opener (2 minutes)",
        target_words: Optional[int] = None,
    ) -> PitchReview:
        """Critique ``pitch_text`` and return a rewritten version.

        Args:
            pitch_text: The founder's raw pitch script.
            company: Optional company name, for context.
            sector: Optional sector, for context.
            stage: Optional stage, for context.
            pitch_format: Which format the rewrite should target.
            target_words: Optional word budget for the rewrite.

        Returns:
            A ``PitchReview`` with ``critique_points`` and ``rewritten_pitch``.

        Raises:
            ValueError: the pitch is empty, too short, or too long.
            PitchEditorConfigurationError: bad key or model.
            PitchEditorError: the call failed or returned nothing usable.
        """
        pitch = (pitch_text or "").strip()
        if not pitch:
            raise ValueError("Paste a pitch script before running a review.")
        if len(pitch) < MIN_PITCH_CHARS:
            raise ValueError(
                f"That is only {len(pitch)} characters. Paste at least "
                f"{MIN_PITCH_CHARS} so there is something to critique."
            )
        if len(pitch) > MAX_PITCH_CHARS:
            raise ValueError(
                f"That pitch is {len(pitch):,} characters, over the "
                f"{MAX_PITCH_CHARS:,} limit. Trim it or review it in sections."
            )

        payload = self._call_model(
            pitch, company, sector, stage, pitch_format, target_words
        )
        review = self._build_review(payload, pitch)
        logger.info(
            "Reviewed pitch: %d critique point(s), %d -> %d words",
            len(review.critique_points),
            review.original_word_count,
            review.rewritten_word_count,
        )
        return review

    # ------------------------------------------------------------------ #
    # Model call
    # ------------------------------------------------------------------ #
    def _call_model(
        self,
        pitch: str,
        company: str,
        sector: str,
        stage: str,
        pitch_format: str,
        target_words: Optional[int],
    ) -> Dict[str, Any]:
        context_lines = [
            f"{label}: {value}"
            for label, value in (
                ("Company", company.strip()),
                ("Sector", sector.strip()),
                ("Stage", stage.strip()),
            )
            if value and value.strip()
        ]
        context_lines.append(f"Target format: {pitch_format}")
        if target_words:
            context_lines.append(f"Target length for the rewrite: ~{target_words} words")

        user_message = (
            "<context>\n" + "\n".join(context_lines) + "\n</context>\n\n"
            "<pitch_script>\n" + pitch + "\n</pitch_script>\n\n"
            "Critique this pitch script the way you would in a partner meeting, "
            "then rewrite it. Return the critique and the rewrite as JSON."
        )

        try:
            return llm.call_tool(
                client=self._client,
                model=self.model,
                system=SYSTEM_PROMPT,
                user_message=user_message,
                tool=REVIEW_TOOL,
                max_tokens=MAX_OUTPUT_TOKENS,
                max_retries=self.max_retries,
            )
        except LLMConfigurationError as exc:
            raise PitchEditorConfigurationError(str(exc)) from exc
        except PitchEditorError:
            raise
        except LLMError as exc:
            raise PitchEditorError(str(exc)) from exc

    # ------------------------------------------------------------------ #
    # Result assembly
    # ------------------------------------------------------------------ #
    def _build_review(self, payload: Dict[str, Any], pitch: str) -> PitchReview:
        """Normalise the model's output into a ``PitchReview``."""
        rewritten = str(payload.get("rewritten_pitch") or "").strip()
        raw_points = payload.get("critique_points")

        points: List[CritiquePoint] = []
        for item in raw_points or []:
            # The schema asks for objects, but tolerate a bare string.
            if isinstance(item, str):
                text = item.strip()
                if text:
                    points.append(CritiquePoint(flaw=text))
                continue
            if not isinstance(item, dict):
                continue
            flaw = str(item.get("flaw") or item.get("issue") or "").strip()
            if not flaw:
                continue
            points.append(
                CritiquePoint(
                    flaw=llm.truncate(flaw, 400),
                    why_it_matters=llm.truncate(item.get("why_it_matters") or "", 600),
                    fix=llm.truncate(item.get("fix") or "", 600),
                    severity=item.get("severity", "medium"),
                )
            )

        if not points and not rewritten:
            raise PitchEditorError(
                "The review came back empty. Try again, or shorten the pitch."
            )
        if not rewritten:
            raise PitchEditorError(
                "The critique came back without a rewritten pitch. Try again."
            )

        points.sort(key=lambda p: p.sort_key)
        return PitchReview(
            critique_points=points,
            rewritten_pitch=rewritten,
            original_pitch=pitch,
            model=self.model,
        )


__all__ = [
    "PitchEditor",
    "PitchReview",
    "CritiquePoint",
    "PitchEditorError",
    "PitchEditorConfigurationError",
    "PITCH_FORMATS",
    "SEVERITIES",
    "MIN_PITCH_CHARS",
    "MAX_PITCH_CHARS",
]
