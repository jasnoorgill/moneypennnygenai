"""Document Q&A screen.

Upload PDFs, then ask questions about them in a chat. Extraction and the model
call live in ``services.qa_engine``; this module only handles widgets and state.
"""

from __future__ import annotations

import streamlit as st

from services.qa_engine import (
    Answer,
    DocumentQAEngine,
    PdfDocument,
    PdfExtractionError,
    extract_text_from_pdf,
    QAEngineConfigurationError,
    QAEngineError,
)

_DOCS_KEY = "qa_documents"       # {upload key: PdfDocument}
_MESSAGES_KEY = "qa_messages"    # [{"role", "content", "caption"}]

SUGGESTIONS = [
    "What are the revenue numbers, and what period do they cover?",
    "What does the document claim about the market size, and how is it derived?",
    "Which claims are asserted without evidence?",
    "What is missing that I would need before an investment committee?",
]


@st.cache_resource(show_spinner=False)
def _engine() -> DocumentQAEngine:
    return DocumentQAEngine()


def _upload_key(uploaded) -> str:
    """Stable identity for an upload, so a rerun does not re-extract it."""
    return f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def _sync_uploads(uploaded_files) -> dict[str, PdfDocument]:
    """Extract anything new, drop anything removed, and return the live set."""
    store: dict[str, PdfDocument] = st.session_state.setdefault(_DOCS_KEY, {})
    current = {_upload_key(f): f for f in (uploaded_files or [])}

    for stale in [key for key in store if key not in current]:
        store.pop(stale)

    for key, uploaded in current.items():
        if key in store:
            continue
        try:
            with st.spinner(f"Reading {uploaded.name}…"):
                store[key] = extract_text_from_pdf(uploaded, filename=uploaded.name)
        except PdfExtractionError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - surface, never crash the page
            st.error(f"Could not read {uploaded.name}: {exc}")

    return store


def _render_document_summary(documents: list[PdfDocument]) -> None:
    if not documents:
        return
    pages = sum(doc.pages_with_text for doc in documents)
    st.caption(
        f"{len(documents)} document(s) loaded · {pages} page(s) with readable text"
    )
    for doc in documents:
        blank = doc.page_count - doc.pages_with_text
        note = f" · {blank} page(s) had no text layer" if blank else ""
        st.caption(f"📄 {doc.filename} — {doc.page_count} page(s){note}")


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def _render_history() -> None:
    for message in st.session_state.get(_MESSAGES_KEY, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("caption"):
                st.caption(message["caption"])


def _answer_caption(answer: Answer) -> str:
    if answer.truncated:
        return (
            f"Answered from the {answer.pages_used} most relevant of "
            f"{answer.pages_total} pages."
        )
    return f"Answered from {answer.pages_total} page(s)."


def _handle_question(question: str, documents: list[PdfDocument]) -> None:
    messages = st.session_state.setdefault(_MESSAGES_KEY, [])
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    engine = _engine()
    engine.sync(documents)
    history = [
        {"role": m["role"], "content": m["content"]} for m in messages[:-1]
    ]

    with st.chat_message("assistant"):
        try:
            with st.spinner("Reading the documents…"):
                answer = engine.ask(question, history=history)
        except QAEngineConfigurationError as exc:
            st.error(f"Configuration problem: {exc}")
            messages.pop()
            return
        except ValueError as exc:
            st.warning(str(exc))
            messages.pop()
            return
        except QAEngineError as exc:
            st.error(f"Could not answer that. {exc}")
            messages.pop()
            return

        st.markdown(answer.text)
        caption = _answer_caption(answer)
        st.caption(caption)

    messages.append(
        {"role": "assistant", "content": answer.text, "caption": caption}
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render() -> None:
    """Render the document Q&A screen."""
    st.title("Document Q&A")
    st.caption(
        "Upload a deck, memo or data-room PDF and interrogate it. Answers come "
        "from the document text only — nothing is filled in from outside it."
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Text-based PDFs only. Scans need OCR first.",
    )

    store = _sync_uploads(uploaded_files)
    documents = list(store.values())

    header, actions = st.columns([4, 1])
    with header:
        _render_document_summary(documents)
    with actions:
        if st.session_state.get(_MESSAGES_KEY) and st.button("Clear chat"):
            st.session_state[_MESSAGES_KEY] = []
            st.rerun()

    if not documents:
        st.info("Upload at least one PDF to start asking questions.")
        _render_history()
        return

    if not st.session_state.get(_MESSAGES_KEY):
        st.markdown("**Try asking:**")
        for suggestion in SUGGESTIONS:
            st.caption(f"· {suggestion}")

    _render_history()

    question = st.chat_input("Ask something about the uploaded document(s)")
    if question:
        _handle_question(question.strip(), documents)
