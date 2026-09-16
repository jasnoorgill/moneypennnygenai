# CLAUDE.md

Moneypenny is a reusable Streamlit application for VC fundraising. It uses an MVC-like architecture. All UI code lives in `ui/`, all business logic and external API calls live in `services/`, and all database connections live in `core/`. We use Supabase for persistent data storage.

## Structure

- `app.py` — entry point: page config, sidebar navigation, dispatch to views.
- `ui/` — Streamlit views. No API calls, no SQL, no `os.getenv`.
  - `match_view.py` (Match) · `editor_view.py` (Editor) · `briefing_view.py` (Briefing) · `qa_view.py` (Q&A)
- `services/` — business logic and external API calls.
  - `llm.py` — shared Gemini plumbing: client, retry/backoff, JSON-schema calls, plain-text calls.
  - `investor_matcher.py` — rank investors against a startup profile.
  - `pitch_editor.py` — critique and rewrite a pitch script.
  - `briefing_service.py` — Tavily research → Gemini briefing → Supabase cache.
  - `qa_engine.py` — PDF extraction (PyPDF2) and grounded document Q&A.
- `core/` — database connections and configuration.
  - `config.py` — loads `.env` once and resolves every environment variable.
  - `database.py` — Supabase client and `DBClient` data access.

## Conventions

- Environment variables are read through `core.config`, never `os.getenv` directly,
  so `.env` is loaded before the first read regardless of import order.
- Services raise their own typed errors (all subclassing `services.llm.LLMError`
  or `core.database.DatabaseError`); views catch those and render a message.
- Model output that names real records is validated against those records before
  it reaches the user — see `_build_matches` in `investor_matcher.py`.
- See `.env.example` for the required keys.
