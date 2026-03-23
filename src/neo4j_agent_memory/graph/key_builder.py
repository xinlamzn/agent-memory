"""Canonical key builders for Memory Store backends.

These functions produce deterministic, human-readable keys for
Memory Store documents.  Neo4j uses UUIDs natively and does not
need these, but they are available for any backend that prefers
structured string keys.

Naming convention: ``{family}::{discriminator}``
"""

from __future__ import annotations

from uuid import UUID


def conversation_key(session_id: str) -> str:
    """Key for a Conversation node."""
    return f"conversation::{session_id}"


def message_key(message_id: str | UUID) -> str:
    """Key for a Message node."""
    return f"message::{message_id}"


def entity_key(name: str, entity_type: str) -> str:
    """Key for an Entity node (name + type for uniqueness)."""
    normalized_name = name.strip().lower().replace(" ", "_")
    normalized_type = entity_type.strip().upper()
    return f"entity::{normalized_name}::{normalized_type}"


def preference_key(preference_id: str | UUID) -> str:
    """Key for a Preference node."""
    return f"preference::{preference_id}"


def fact_key(fact_id: str | UUID) -> str:
    """Key for a Fact node."""
    return f"fact::{fact_id}"


def trace_key(trace_id: str | UUID) -> str:
    """Key for a ReasoningTrace node."""
    return f"trace::{trace_id}"


def step_key(step_id: str | UUID) -> str:
    """Key for a ReasoningStep node."""
    return f"step::{step_id}"


def tool_key(tool_name: str) -> str:
    """Key for a Tool node."""
    return f"tool::{tool_name}"


def tool_call_key(tool_call_id: str | UUID) -> str:
    """Key for a ToolCall node."""
    return f"toolcall::{tool_call_id}"
