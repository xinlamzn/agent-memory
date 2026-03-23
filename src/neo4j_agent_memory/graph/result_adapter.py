"""Shared adapters from backend result shapes to package models.

Both the Neo4j and Memory Store backends return raw dicts from their
operations.  This module centralises the conversion from those dicts
to the Pydantic models used in the public API so that each backend
does not have to duplicate the parsing logic.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_python_datetime(value: Any) -> datetime:
    """Coerce a backend datetime value to a Python datetime.

    Handles:
    - ``None`` -> ``datetime.utcnow()``
    - Python ``datetime`` -> pass-through
    - Neo4j ``DateTime`` (has ``.to_native()``) -> native datetime
    - ISO-8601 strings -> ``datetime.fromisoformat``
    """
    if value is None:
        return datetime.utcnow()
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.utcnow()
    try:
        return value.to_native()  # Neo4j DateTime
    except AttributeError:
        return datetime.utcnow()


def deserialize_metadata(raw: str | dict | None) -> dict[str, Any]:
    """Deserialize metadata from a JSON string or passthrough dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def serialize_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Serialize a metadata dict to a JSON string for storage."""
    if not metadata:
        return None
    return json.dumps(metadata)


def safe_uuid(value: str | UUID | None) -> UUID | None:
    """Coerce a string to UUID safely."""
    if value is None:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return None
