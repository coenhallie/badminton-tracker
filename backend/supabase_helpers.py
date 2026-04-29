"""
Shared helpers for Modal apps that talk to Supabase.

Used by:
- backend/modal_supabase_processor.py
- backend/modal_pdf_export.py
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import os

_supabase_client = None


def supabase_client():
    """Return a singleton Supabase client authenticated with the service-role key.

    Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from environment
    (loaded from the 'supabase-secrets' Modal Secret on each function).
    """
    global _supabase_client
    if _supabase_client is None:
        from supabase import create_client
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        _supabase_client = create_client(url, key)
    return _supabase_client


def verify_hmac(body: bytes, signature: str | None, secret: str) -> bool:
    """Verify HMAC-SHA256 of body matches signature (hex-encoded).

    Returns False on any missing input — callers must treat as auth failure.
    """
    if not signature or not secret:
        return False
    expected = _hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return _hmac.compare_digest(expected, signature)
