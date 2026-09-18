# Running Moneypenny

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Get the three keys

**Supabase** — https://supabase.com → sign in → New project (free tier is fine;
it takes a couple of minutes to provision). Then **Settings → API Keys**:
- `SUPABASE_URL` is the Project URL (`https://<ref>.supabase.co`)
- `SUPABASE_KEY` is a **secret** key (`sb_secret_...`). The older `service_role`
  JWT still works — Supabase is deprecating `anon`/`service_role` by the end of
  2026 — so prefer the new secret key on a new project.

The secret key bypasses row-level security, which is what you want for a
server-side app like this one, and is exactly why it must never reach the
browser or a public repo.

**Google Gemini** — https://aistudio.google.com/apikey → sign in with a Google
account → Create API key. Copy it immediately. Gemini has a free tier, so the
key usually works straight away without adding billing; enable billing only if
you hit the free-tier rate limits.

**Tavily** — https://app.tavily.com → sign up (no card needed) → copy the key
from the dashboard. The free tier is 1,000 credits/month; this app uses 3
advanced searches per fresh briefing, so 2 credits each = 6 credits per
briefing, and cached briefings cost nothing.

**MiniMax** (optional, alternative to Gemini) — https://www.minimax.io →
sign in → API Keys → create a key. Outside mainland China the app talks to
MiniMax's Anthropic-compatible Messages API at
`https://api.minimax.io/anthropic/v1/messages` (mainland China accounts use
`api.minimaxi.com` instead — set `MINIMAX_BASE_URL` to override). Set
`LLM_PROVIDER=minimax` to use it instead of Gemini.

## 3. Add them to a .env

Create a `.env` file in the repo root (same folder as `app.py`). It is
gitignored, so your keys never get committed.

```bash
cat > .env <<'KEYS'
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
GEMINI_API_KEY=AIza...
TAVILY_API_KEY=tvly-...
MINIMAX_API_KEY=your-minimax-key
KEYS
```

Restart the app after creating it — `.env` is read once at startup, so a
browser refresh alone will not pick up new keys.

### On GitHub Codespaces

`.env` works, but it disappears if the Codespace is rebuilt. For something
durable use Codespaces secrets instead:

GitHub → Settings → Codespaces → Secrets → New secret, one per key, scoped to
this repository. Then rebuild the Codespace (Cmd/Ctrl+Shift+P → "Codespaces:
Rebuild Container"). They arrive as environment variables, which the app reads
without any `.env` at all.

### On Streamlit Community Cloud

Paste the same keys into Settings → Secrets in TOML form:

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-service-role-key"
GEMINI_API_KEY = "AIza..."
TAVILY_API_KEY = "tvly-..."
```

## 4. Create the Supabase tables

Run this in the Supabase SQL editor.

```sql
create table if not exists investors (
    id          bigint primary key generated always as identity,
    name        text not null,
    firm        text,
    sectors     text,
    stages      text,
    check_size  text,
    geography   text,
    thesis      text,
    website     text,
    created_at  timestamptz not null default now()
);

create table if not exists partner_briefings (
    id            bigint primary key generated always as identity,
    partner_name  text not null unique,
    briefing      jsonb not null,
    sources       jsonb,
    model         text,
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);

create index if not exists partner_briefings_name_idx
    on partner_briefings (lower(partner_name));
```

Column names are flexible on `investors` — the matcher also recognises
`investor_name`, `focus`, `verticals`, `stage`, `cheque_size`, `ticket_size`,
`location`, `region`, `notes` and others, so an existing sheet usually imports
as-is.

If you have row-level security on, either use the service-role key or add a
policy that lets the anon role read `investors` and read/write
`partner_briefings`.

## 5. Run

```bash
streamlit run app.py
```

The sidebar shows which services are still unconfigured. When it reads
"✅ All services configured", every view is live.

## What needs what

| View     | Needs                          |
|----------|--------------------------------|
| Match    | Supabase + Gemini              |
| Editor   | Gemini                         |
| Briefing | Supabase + Gemini + Tavily     |
| Q&A      | Gemini                         |

Editor and Q&A work with just `GEMINI_API_KEY` — good for a first smoke test
before the database is set up.

## If you see "Gemini is busy right now"

That is a 503 from Google: the model is at capacity, not a problem with your key
or your input. The app already retries four times with backoff and then tries
`GEMINI_FALLBACK_MODELS` in order, so you only see this message when every model
in the chain is busy. Wait a minute and run it again.

If the newest model is busy often, make a steadier one the default:

```
GEMINI_MODEL=gemini-2.5-flash
```

`gemini-2.5-flash` is older and less in demand, and it is more than capable for
everything Moneypenny asks of it.
