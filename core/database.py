"""Supabase connection and data access for Moneypenny.

All database access goes through this module. Nothing here imports Streamlit or
calls an external API other than Supabase — business logic belongs in
``services/`` and presentation in ``ui/``.

Environment variables
---------------------
SUPABASE_URL                 (required) e.g. https://xxxx.supabase.co
SUPABASE_KEY                 (required) service-role or anon key.
                             ``SUPABASE_SERVICE_KEY`` / ``SUPABASE_ANON_KEY``
                             are accepted as fallbacks.
SUPABASE_INVESTORS_TABLE     (optional) defaults to "investors"
SUPABASE_BRIEFINGS_TABLE     (optional) defaults to "partner_briefings"

Assumed schema
--------------
investors(id, name, firm, ... , created_at)

partner_briefings(
    id             bigint primary key generated always as identity,
    partner_name   text not null unique,
    briefing       jsonb not null,
    sources        jsonb,
    model          text,
    created_at     timestamptz not null default now(),
    updated_at     timestamptz not null default now()
)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from postgrest.exceptions import APIError
from supabase import Client, create_client

from core import config

logger = logging.getLogger(__name__)

INVESTORS_TABLE = config.env("SUPABASE_INVESTORS_TABLE", default="investors")
BRIEFINGS_TABLE = config.env("SUPABASE_BRIEFINGS_TABLE", default="partner_briefings")

# Supabase caps a single PostgREST response (1000 rows by default), so full
# table reads are paged.
_PAGE_SIZE = 1000


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class DatabaseError(Exception):
    """Base class for every failure raised by this module."""


class ConfigurationError(DatabaseError):
    """Credentials or other required configuration are missing/invalid."""


class QueryError(DatabaseError):
    """A query reached Supabase but failed (bad table, RLS, constraint, ...)."""


class RecordNotFoundError(DatabaseError):
    """A record that the caller required does not exist."""


# --------------------------------------------------------------------------- #
# Client factory
# --------------------------------------------------------------------------- #
_client: Optional[Client] = None
_client_lock = threading.Lock()


def _read_credentials() -> tuple[str, str]:
    url = config.supabase_url()
    key = config.supabase_key()

    missing = [
        name
        for name, value in (("SUPABASE_URL", url), ("SUPABASE_KEY", key))
        if not value
    ]
    if missing:
        raise ConfigurationError(
            "Missing environment variable(s): "
            + ", ".join(missing)
            + ". Set them in your environment or .env file before starting the app."
        )
    if not url.startswith(("http://", "https://")):
        raise ConfigurationError(
            f"SUPABASE_URL must start with https:// or http:// (got {url!r})."
        )
    return url, key


def get_client(refresh: bool = False) -> Client:
    """Return a process-wide Supabase client, creating it on first use.

    Raises:
        ConfigurationError: credentials are missing or the client cannot be built.
    """
    global _client
    if _client is not None and not refresh:
        return _client

    with _client_lock:
        if _client is not None and not refresh:
            return _client
        url, key = _read_credentials()
        try:
            _client = create_client(url, key)
        except Exception as exc:  # network, bad key format, library errors
            raise ConfigurationError(
                f"Could not initialise the Supabase client: {exc}"
            ) from exc
        logger.info("Supabase client initialised for %s", url)
        return _client


def _normalise_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("partner_name must be a non-empty string.")
    return " ".join(name.split())


def _escape_like(value: str) -> str:
    """Escape PostgREST/SQL LIKE wildcards so a name is matched literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
class DBClient:
    """Thin, typed wrapper over the Supabase tables Moneypenny uses."""

    def __init__(self, client: Optional[Client] = None) -> None:
        # Passing a client in keeps this class trivially testable.
        self._client = client or get_client()

    @property
    def client(self) -> Client:
        return self._client

    # ---------------------------------------------------------------- #
    # 1. Investors
    # ---------------------------------------------------------------- #
    def fetch_all_investors(
        self,
        columns: str = "*",
        order_by: str = "name",
        descending: bool = False,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch every investor row, paging past Supabase's per-request cap.

        Args:
            columns: PostgREST column selection, e.g. "id,name,firm".
            order_by: Column to sort on. Must exist on the table; pass ""
                to leave the order to PostgREST.
            descending: Sort direction.
            filters: Optional equality filters, e.g. {"stage": "seed"}.
            limit: Stop after this many rows (None = all rows).

        Returns:
            A list of row dicts; an empty list if the table has no rows.

        Raises:
            QueryError: the table is missing, RLS denied access, or the
                request failed.
        """
        rows: List[Dict[str, Any]] = []
        offset = 0

        try:
            while True:
                page_size = _PAGE_SIZE
                if limit is not None:
                    remaining = limit - len(rows)
                    if remaining <= 0:
                        break
                    page_size = min(page_size, remaining)

                query = self._client.table(INVESTORS_TABLE).select(columns)
                for column, value in (filters or {}).items():
                    query = (
                        query.is_(column, "null")
                        if value is None
                        else query.eq(column, value)
                    )
                if order_by:
                    query = query.order(order_by, desc=descending)
                query = query.range(offset, offset + page_size - 1)

                response = query.execute()
                page: Sequence[Dict[str, Any]] = response.data or []
                rows.extend(page)

                if len(page) < page_size:
                    break
                offset += page_size

        except APIError as exc:
            logger.exception("Failed to fetch investors")
            raise QueryError(
                f"Could not read table '{INVESTORS_TABLE}': "
                f"{exc.message or exc}"
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error fetching investors")
            raise QueryError(f"Unexpected error fetching investors: {exc}") from exc

        logger.info("Fetched %d investor(s) from %s", len(rows), INVESTORS_TABLE)
        return rows

    # ---------------------------------------------------------------- #
    # 2. Cache a partner briefing
    # ---------------------------------------------------------------- #
    def insert_partner_briefing(
        self,
        partner_name: str,
        briefing: Any,
        sources: Optional[Any] = None,
        model: Optional[str] = None,
        overwrite: bool = True,
    ) -> Dict[str, Any]:
        """Write a generated partner briefing into the cache.

        By default this upserts on ``partner_name`` so regenerating a briefing
        replaces the cached copy instead of raising a unique-constraint error.

        Args:
            partner_name: Partner the briefing is about. Whitespace-normalised.
            briefing: The briefing payload (dict or str).
            sources: Optional list/dict of citations used to build it.
            model: Optional identifier of the model that produced it.
            overwrite: Upsert when True; plain insert (fails on duplicates)
                when False.

        Returns:
            The stored row.

        Raises:
            ValueError: ``partner_name`` or ``briefing`` is empty.
            QueryError: the write failed, including a duplicate key when
                ``overwrite`` is False.
        """
        name = _normalise_name(partner_name)
        if briefing is None or (isinstance(briefing, str) and not briefing.strip()):
            raise ValueError("briefing must not be empty.")

        payload: Dict[str, Any] = {
            "partner_name": name,
            "briefing": briefing,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if sources is not None:
            payload["sources"] = sources
        if model:
            payload["model"] = model

        try:
            table = self._client.table(BRIEFINGS_TABLE)
            if overwrite:
                response = table.upsert(
                    payload, on_conflict="partner_name"
                ).execute()
            else:
                response = table.insert(payload).execute()
        except APIError as exc:
            logger.exception("Failed to cache briefing for %s", name)
            if getattr(exc, "code", None) == "23505":
                raise QueryError(
                    f"A briefing for '{name}' already exists. "
                    "Call with overwrite=True to replace it."
                ) from exc
            raise QueryError(
                f"Could not write to '{BRIEFINGS_TABLE}': {exc.message or exc}"
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error caching briefing for %s", name)
            raise QueryError(f"Unexpected error caching briefing: {exc}") from exc

        data = response.data or []
        if not data:
            raise QueryError(
                f"Briefing for '{name}' was not returned by Supabase after the "
                "write; check the table's RLS policies."
            )
        logger.info("Cached briefing for %s", name)
        return data[0]

    # ---------------------------------------------------------------- #
    # 3. Read a partner briefing back
    # ---------------------------------------------------------------- #
    def get_partner_briefing(
        self,
        partner_name: str,
        max_age_hours: Optional[float] = None,
        required: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Return the cached briefing for a partner, if there is a fresh one.

        The lookup is case-insensitive and whitespace-tolerant.

        Args:
            partner_name: Partner to look up.
            max_age_hours: Treat anything older than this as a cache miss.
                None disables the freshness check.
            required: Raise ``RecordNotFoundError`` instead of returning None.

        Returns:
            The row dict, or None on a miss (when ``required`` is False).

        Raises:
            ValueError: ``partner_name`` is empty.
            RecordNotFoundError: nothing usable found and ``required`` is True.
            QueryError: the read failed.
        """
        name = _normalise_name(partner_name)

        try:
            response = (
                self._client.table(BRIEFINGS_TABLE)
                .select("*")
                .ilike("partner_name", _escape_like(name))
                .order("updated_at", desc=True)
                .limit(1)
                .execute()
            )
        except APIError as exc:
            logger.exception("Failed to read briefing for %s", name)
            raise QueryError(
                f"Could not read '{BRIEFINGS_TABLE}': {exc.message or exc}"
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error reading briefing for %s", name)
            raise QueryError(f"Unexpected error reading briefing: {exc}") from exc

        rows = response.data or []
        if not rows:
            logger.info("Cache miss: no briefing stored for %s", name)
            if required:
                raise RecordNotFoundError(f"No briefing cached for '{name}'.")
            return None

        row = rows[0]
        if max_age_hours is not None and not self._is_fresh(row, max_age_hours):
            logger.info("Cache miss: briefing for %s is stale", name)
            if required:
                raise RecordNotFoundError(
                    f"The cached briefing for '{name}' is older than "
                    f"{max_age_hours} hour(s)."
                )
            return None

        return row

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #
    @staticmethod
    def _is_fresh(row: Dict[str, Any], max_age_hours: float) -> bool:
        """True if the row's timestamp is within ``max_age_hours`` of now.

        An unparseable or missing timestamp is treated as stale so the caller
        regenerates rather than serving something of unknown age.
        """
        raw = row.get("updated_at") or row.get("created_at")
        if not raw:
            return False
        try:
            stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Unparseable timestamp on cached briefing: %r", raw)
            return False
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - stamp <= timedelta(hours=max_age_hours)


__all__ = [
    "DBClient",
    "get_client",
    "DatabaseError",
    "ConfigurationError",
    "QueryError",
    "RecordNotFoundError",
    "INVESTORS_TABLE",
    "BRIEFINGS_TABLE",
]
