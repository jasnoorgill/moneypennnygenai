"""Document Q&A over uploaded PDFs.

Two functions do the work — ``extract_text_from_pdf`` pulls text out of a PDF
with PyPDF2, and ``ask_document_question`` sends that text plus the user's
question to Gemini — and ``DocumentQAEngine`` holds the document context
between turns so the UI only has to pass the question.

The answering prompt is deliberately strict: the model answers from the
supplied text alone and says so when the document does not cover something.

Environment variables
---------------------
GEMINI_API_KEY      (required)
GEMINI_MODEL        (optional)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

try:  # PyPDF2 3.x is the pinned dependency; pypdf is the same library renamed.
    from PyPDF2 import PdfReader
    from PyPDF2.errors import PdfReadError
except ImportError:  # pragma: no cover
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

from services import llm
from services.llm import (
    DEFAULT_MODEL,
    MAX_RETRIES,
    Client,
    LLMConfigurationError,
    LLMError,
)

logger = logging.getLogger(__name__)

# Characters of document text sent per question. Roughly 45k tokens — large
# enough for a full deck or data room memo, small enough to stay affordable.
MAX_CONTEXT_CHARS = 180_000
MAX_ANSWER_TOKENS = 1500
MAX_HISTORY_TURNS = 8
MIN_USEFUL_CHARS = 200

PdfSource = Union[str, Path, bytes, bytearray, Any]


class QAError(Exception):
    """Base class for failures in this module."""


class PdfExtractionError(QAError):
    """The PDF could not be read, or holds no extractable text."""


class QAEngineError(QAError, LLMError):
    """The question could not be answered."""


class QAEngineConfigurationError(QAEngineError, LLMConfigurationError):
    """The Gemini API key or model configuration is missing/invalid."""


# --------------------------------------------------------------------------- #
# Document objects
# --------------------------------------------------------------------------- #
@dataclass
class PdfDocument:
    """Extracted text from one PDF, kept page by page so answers can cite pages."""

    filename: str
    pages: List[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def pages_with_text(self) -> int:
        return sum(1 for page in self.pages if page.strip())

    @property
    def text(self) -> str:
        """The whole document with page markers, ready to hand to the model."""
        return "\n\n".join(
            f"[page {number}]\n{content}"
            for number, content in enumerate(self.pages, start=1)
            if content.strip()
        )

    @property
    def char_count(self) -> int:
        return sum(len(page) for page in self.pages)


@dataclass
class Answer:
    """One answer, with the provenance needed to check it."""

    text: str
    documents: List[str] = field(default_factory=list)
    pages_used: int = 0
    pages_total: int = 0

    @property
    def truncated(self) -> bool:
        return 0 < self.pages_used < self.pages_total


# --------------------------------------------------------------------------- #
# 1. Extraction
# --------------------------------------------------------------------------- #
def extract_text_from_pdf(source: PdfSource, filename: str = "") -> PdfDocument:
    """Extract text from a PDF using PyPDF2.

    Args:
        source: A path, raw bytes, or any file-like object with ``.read()`` —
            including a Streamlit ``UploadedFile``.
        filename: Display name. Inferred from a path or an uploaded file's
            ``.name`` when omitted.

    Returns:
        A ``PdfDocument`` whose ``pages`` list holds one string per page.

    Raises:
        PdfExtractionError: the file is unreadable, password-protected, or has
            no extractable text (a scan, typically).
    """
    name = filename or _infer_name(source)

    try:
        stream = _as_stream(source)
        reader = PdfReader(stream)
    except PdfReadError as exc:
        raise PdfExtractionError(
            f"'{name}' is not a readable PDF ({exc}). Re-export it and try again."
        ) from exc
    except FileNotFoundError as exc:
        raise PdfExtractionError(f"Could not find '{name}'.") from exc
    except Exception as exc:
        raise PdfExtractionError(f"Could not open '{name}': {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            # Many decks are "encrypted" with an empty owner password.
            if not reader.decrypt(""):
                raise PdfExtractionError(
                    f"'{name}' is password-protected. Remove the password and "
                    "upload it again."
                )
        except PdfExtractionError:
            raise
        except Exception as exc:
            raise PdfExtractionError(
                f"'{name}' is password-protected and could not be opened: {exc}"
            ) from exc

    pages: List[str] = []
    failed_pages: List[int] = []
    for index, page in enumerate(getattr(reader, "pages", []), start=1):
        try:
            pages.append(_clean(page.extract_text() or ""))
        except Exception as exc:  # one bad page should not lose the document
            logger.warning("Could not extract page %d of %s: %s", index, name, exc)
            pages.append("")
            failed_pages.append(index)

    document = PdfDocument(filename=name, pages=pages)

    if not document.page_count:
        raise PdfExtractionError(f"'{name}' has no pages.")
    if document.char_count < MIN_USEFUL_CHARS:
        raise PdfExtractionError(
            f"Almost no text could be read from '{name}'. If it is a scan or an "
            "image-only export, run OCR on it first — PyPDF2 reads text layers, "
            "not pictures of text."
        )
    if failed_pages:
        logger.info("Skipped %d unreadable page(s) in %s", len(failed_pages), name)

    logger.info(
        "Extracted %s: %d page(s), %d with text, %d chars",
        name, document.page_count, document.pages_with_text, document.char_count,
    )
    return document


def _infer_name(source: PdfSource) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).name
    return getattr(source, "name", "document.pdf")


def _as_stream(source: PdfSource) -> Any:
    if isinstance(source, (bytes, bytearray)):
        import io

        return io.BytesIO(source)
    if isinstance(source, Path):
        return str(source)
    if hasattr(source, "seek"):
        try:
            source.seek(0)  # an uploaded file may already have been read
        except Exception:
            pass
    return source


def _clean(text: str) -> str:
    """Normalise PDF text: de-hyphenate line breaks, collapse blank runs."""
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# 2. Answering
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a skeptical venture capital investor reading a founder's "
    "documents during diligence. You are answering questions for a colleague "
    "who needs to know what the documents actually say.\n\n"
    "Grounding rules — these override everything else:\n"
    "- Answer ONLY from the document text supplied in this conversation. Your "
    "own knowledge of the company, its market, or its competitors is not "
    "evidence and must not appear in an answer.\n"
    "- If the documents do not answer the question, say plainly that the "
    "document does not cover it, and name what a founder would need to supply. "
    "Never fill a gap with a plausible guess.\n"
    "- Cite the page you are drawing on, like [page 4], for every substantive "
    "claim. If pages disagree, say so and cite both.\n"
    "- Quote the document's own wording for any number, date, or named "
    "customer rather than paraphrasing it.\n\n"
    "How you read:\n"
    "- Distinguish what the document demonstrates from what it merely asserts. "
    "'We are the market leader' with no evidence behind it is a claim, not a "
    "fact, and you say so.\n"
    "- Flag the diligence gap when you see one: a metric with no time period, "
    "a growth rate with no base, a pipeline counted as revenue, a TAM with no "
    "derivation, an unexplained definitional change between pages.\n"
    "- Be concise and specific. No praise, no hedging, no restating the "
    "question. Lead with the answer, then the evidence.\n"
    "- Skepticism means rigour about evidence, not cynicism: where the "
    "documents do support a claim, say so directly."
)


def build_context(
    documents: Sequence[PdfDocument],
    question: str = "",
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, int, int]:
    """Assemble document text for the prompt, trimming to ``max_chars``.

    When the documents are too large, pages are ranked by keyword overlap with
    the question and the best-scoring ones are kept in their original order, so
    a 200-page data room still answers a narrow question.

    Returns:
        ``(context, pages_used, pages_total)``.
    """
    entries: List[tuple[str, int, str]] = [
        (doc.filename, number, content)
        for doc in documents
        for number, content in enumerate(doc.pages, start=1)
        if content.strip()
    ]
    pages_total = len(entries)
    if not entries:
        return "", 0, 0

    def render(chosen: Iterable[tuple[str, int, str]]) -> str:
        blocks: List[str] = []
        current: Optional[str] = None
        for name, number, content in chosen:
            if name != current:
                if current is not None:
                    blocks.append("</document>")
                blocks.append(f'<document name="{name}">')
                current = name
            blocks.append(f"[page {number}]\n{content}")
        if current is not None:
            blocks.append("</document>")
        return "\n\n".join(blocks)

    full = render(entries)
    if len(full) <= max_chars:
        return full, pages_total, pages_total

    keywords = _keywords(question)

    def relevance(entry: tuple[str, int, str]) -> tuple[int, int]:
        overlap = len(keywords & _keywords(entry[2])) if keywords else 0
        return (-overlap, entry[1])  # then prefer earlier pages

    budget = max_chars
    kept: List[tuple[str, int, str]] = []
    for entry in sorted(entries, key=relevance):
        cost = len(entry[2]) + 40
        if cost > budget:
            continue
        budget -= cost
        kept.append(entry)

    if not kept:
        # Every single page is larger than the whole budget (a one-page export
        # of a long memo, typically). Send the most relevant one, cut to fit,
        # rather than sending nothing at all.
        name, number, content = sorted(entries, key=relevance)[0]
        clipped = content[: max(0, max_chars - 60)].rstrip()
        kept = [(name, number, f"{clipped}\n…[page truncated to fit]")]
        logger.info("Single page exceeded the context budget; clipped page %d", number)
        return render(kept), 1, pages_total

    kept.sort(key=lambda e: (e[0], e[1]))
    logger.info("Context trimmed to %d of %d pages", len(kept), pages_total)
    return render(kept), len(kept), pages_total


def ask_document_question(
    document_text: str,
    question: str,
    client: Optional[Client] = None,
    model: str = DEFAULT_MODEL,
    history: Optional[Sequence[Dict[str, str]]] = None,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Send extracted document text and a question to Gemini.

    Args:
        document_text: Text from ``extract_text_from_pdf`` (or ``build_context``).
        question: The user's question.
        client: Gemini client. Built from the environment when omitted.
        model: Model id.
        history: Prior ``{"role", "content"}`` turns, for follow-up questions.
        max_retries: Retries for transient API failures.

    Returns:
        The answer as plain text.

    Raises:
        ValueError: the question or the document text is empty.
        QAEngineConfigurationError: bad key or model.
        QAEngineError: the call failed.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("Ask a question first.")
    if not (document_text or "").strip():
        raise ValueError("No document text to answer from. Upload a PDF first.")

    if client is None:
        client = _build_client()

    messages: List[Dict[str, Any]] = []
    for turn in list(history or [])[-MAX_HISTORY_TURNS * 2 :]:
        role, content = turn.get("role"), (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # Trailing document context keeps the question closest to the answer.
    messages.append(
        {
            "role": "user",
            "content": (
                "<documents>\n"
                f"{document_text}\n"
                "</documents>\n\n"
                f"<question>\n{question}\n</question>\n\n"
                "Answer from the documents above only, citing pages."
            ),
        }
    )

    if messages[0]["role"] != "user":  # the API requires a user turn first
        messages.pop(0)

    try:
        return llm.call_text(
            client=client,
            model=model,
            system=SYSTEM_PROMPT,
            messages=messages,
            max_tokens=MAX_ANSWER_TOKENS,
            max_retries=max_retries,
        )
    except LLMConfigurationError as exc:
        raise QAEngineConfigurationError(str(exc)) from exc
    except QAEngineError:
        raise
    except LLMError as exc:
        raise QAEngineError(str(exc)) from exc


def _build_client() -> Client:
    try:
        return llm.build_client()
    except LLMConfigurationError as exc:
        raise QAEngineConfigurationError(str(exc)) from exc


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "what", "how", "why",
    "does", "did", "are", "was", "were", "their", "there", "from", "have",
    "has", "not", "you", "your", "our", "about", "into", "than", "then",
}


def _keywords(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", (text or "").lower())
        if len(token) > 3 and token not in _STOPWORDS
    }


# --------------------------------------------------------------------------- #
# Engine — holds the document context between questions
# --------------------------------------------------------------------------- #
class DocumentQAEngine:
    """Holds uploaded documents and answers questions against them."""

    def __init__(
        self,
        client: Optional[Client] = None,
        model: str = DEFAULT_MODEL,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.model = model
        self.max_context_chars = max_context_chars
        self.max_retries = max_retries
        self._client = client or _build_client()
        self._documents: Dict[str, PdfDocument] = {}

    # ---------------------------------------------------------------- #
    # Document context
    # ---------------------------------------------------------------- #
    @property
    def documents(self) -> List[PdfDocument]:
        return list(self._documents.values())

    @property
    def filenames(self) -> List[str]:
        return list(self._documents)

    @property
    def is_empty(self) -> bool:
        return not self._documents

    @property
    def total_pages(self) -> int:
        return sum(doc.pages_with_text for doc in self._documents.values())

    def add_pdf(self, source: PdfSource, filename: str = "") -> PdfDocument:
        """Extract a PDF and keep it as context. Re-adding replaces it.

        Raises:
            PdfExtractionError: the PDF is unreadable or has no text.
        """
        document = extract_text_from_pdf(source, filename)
        self._documents[document.filename] = document
        return document

    def add_document(self, document: PdfDocument) -> PdfDocument:
        """Keep an already-extracted document as context."""
        self._documents[document.filename] = document
        return document

    def remove(self, filename: str) -> bool:
        return self._documents.pop(filename, None) is not None

    def clear(self) -> None:
        self._documents.clear()

    def sync(self, documents: Iterable[PdfDocument]) -> None:
        """Replace the whole context with ``documents``."""
        self._documents = {doc.filename: doc for doc in documents}

    # ---------------------------------------------------------------- #
    # Answering
    # ---------------------------------------------------------------- #
    def ask(
        self, question: str, history: Optional[Sequence[Dict[str, str]]] = None
    ) -> Answer:
        """Answer ``question`` from the loaded documents.

        Raises:
            ValueError: no question, or no documents loaded.
            QAEngineConfigurationError: bad key or model.
            QAEngineError: the call failed.
        """
        if self.is_empty:
            raise ValueError("Upload a PDF before asking a question.")

        context, pages_used, pages_total = build_context(
            self.documents, question, self.max_context_chars
        )
        if not context:
            raise ValueError(
                "The uploaded document(s) contain no readable text to answer from."
            )

        text = ask_document_question(
            document_text=context,
            question=question,
            client=self._client,
            model=self.model,
            history=history,
            max_retries=self.max_retries,
        )
        return Answer(
            text=text,
            documents=self.filenames,
            pages_used=pages_used,
            pages_total=pages_total,
        )


__all__ = [
    "DocumentQAEngine",
    "PdfDocument",
    "Answer",
    "extract_text_from_pdf",
    "ask_document_question",
    "build_context",
    "QAError",
    "PdfExtractionError",
    "QAEngineError",
    "QAEngineConfigurationError",
    "MAX_CONTEXT_CHARS",
]
