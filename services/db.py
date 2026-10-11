"""
Supabase backend client for persisted research state.

Owns sessions history (research_sessions / session_messages), saved artifacts
(saved_artifacts), and bookmarked papers (saved_papers). All helpers degrade
gracefully: if the SDK or credentials are missing, clients return None and
endpoints answer 503 without affecting the search/LLM pipelines.
"""

import logging
import os
from typing import Any, Dict, Optional

from fastapi import Header

logger = logging.getLogger("academic_assistant.db")

try:
    from supabase import Client, create_client
    _HAS_SUPABASE_SDK = True
except ImportError:  # pragma: no cover - offline environments
    Client = None  # type: ignore
    create_client = None  # type: ignore
    _HAS_SUPABASE_SDK = False

SUPABASE_URL = ""
SUPABASE_KEY = ""


def _creds() -> tuple:
    """
    Resolve credentials lazily: services/* imports run before load_dotenv()
    in main.py, so module-level reads would miss .env values.
    """
    url = os.getenv("SUPABASE_URL", "").strip()
    key = (
        os.getenv("SUPABASE_KEY", "").strip()
        or os.getenv("SUPABASE_ANON_KEY", "").strip()
    )
    return url, key


def is_configured() -> bool:
    """True when the SDK is installed and URL + publishable key are present."""
    url, key = _creds()
    return bool(_HAS_SUPABASE_SDK and url and key)


_base_client: Optional[Any] = None


def get_supabase_client() -> Optional[Any]:
    """
    Shared anonymous Supabase client (respects RLS as the anon role).
    Returns None when unconfigured; callers must handle None with a 503.
    """
    global _base_client
    if not is_configured():
        return None
    if _base_client is None:
        try:
            url, key = _creds()
            _base_client = create_client(url, key)
        except Exception as e:
            logger.error("Failed to initialize Supabase client: %s", str(e))
            return None
    return _base_client


def get_user_client(access_token: str) -> Optional[Any]:
    """
    Per-request client authenticated as the end user so Row Level Security
    policies (auth.uid() = user_id) apply. Returns None when unconfigured.
    """
    client = get_supabase_client()
    if client is None or not access_token:
        return None
    try:
        client.postgrest.auth(access_token)
        return client
    except Exception as e:
        logger.warning("Failed to bind user token to Supabase client: %s", str(e))
        return None


def _extract_bearer(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


async def get_current_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency parsing `Authorization: Bearer <access_token>` via
    `supabase.auth.get_user(token)`. Returns {"id", "email"} or None when
    unauthenticated/misconfigured — never raises, so guest flows stay intact.
    """
    token = _extract_bearer(authorization)
    if not token:
        return None
    client = get_supabase_client()
    if client is None:
        return None
    try:
        resp = client.auth.get_user(token)
        user = getattr(resp, "user", None)
        if user is None and isinstance(resp, dict):
            user = resp.get("user")
        if user is None:
            return None
        uid = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
        email = getattr(user, "email", None) or (user.get("email") if isinstance(user, dict) else None)
        if not uid:
            return None
        return {"id": str(uid), "email": email, "token": token}
    except Exception as e:
        logger.debug("Supabase token validation failed: %s", str(e))
        return None
