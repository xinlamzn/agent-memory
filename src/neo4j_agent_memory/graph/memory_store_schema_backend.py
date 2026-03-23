"""Memory Store implementation of the SchemaBackend protocol.

The Memory Store auto-creates indices on first write, so most schema
management methods are intentional no-ops.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neo4j_agent_memory.graph.backend_protocol import UnsupportedBackendOperation

if TYPE_CHECKING:
    from neo4j_agent_memory.graph.memory_store_backend import MemoryStoreGraphBackend

logger = logging.getLogger(__name__)


class MemoryStoreSchemaBackend:
    """``SchemaBackend`` implementation for the Memory Store.

    Indices are auto-created and auto-managed by the OpenSearch Graph
    Plugin on first write.  Vector (kNN) mappings are created when the
    first embedding is upserted.  Constraints are unnecessary because
    node identity is deterministic (SHA-256 of scope + key).
    """

    def __init__(self, graph: MemoryStoreGraphBackend) -> None:
        self._graph = graph

    async def setup_all(self) -> None:
        """No-op: indices are auto-created on first write."""

    async def setup_constraints(self) -> None:
        """No-op: deterministic IDs provide uniqueness."""

    async def setup_indexes(self) -> None:
        """No-op: auto-managed by the Memory Store."""

    async def setup_vector_indexes(self) -> None:
        """No-op: kNN mapping created on first embedding upsert."""

    async def setup_point_indexes(self) -> None:
        """Geospatial point indexes are not supported by the Memory Store."""
        raise UnsupportedBackendOperation(
            "setup_point_indexes (geospatial)",
            "memory_store",
            hint="Geospatial queries are only available with backend='neo4j'.",
        )

    async def drop_all(self) -> None:
        """Delete all memory data by paginating and deleting each node.

        Uses ``_query`` enumerate mode with ``search_after`` pagination
        to discover all nodes, then ``_delete`` to remove each one.
        """
        from neo4j_agent_memory.graph.memory_store_backend import LABEL_NAMESPACE

        for label, namespace in LABEL_NAMESPACE.items():
            body: dict[str, Any] = {
                "tenant_id": self._graph._config.tenant_id,
                "user_id": self._graph._config.user_id,
                "namespace": namespace,
                "labels": [label],
                "top_k": 500,
            }

            while True:
                result = await self._graph._post("_query", body)
                if not result:
                    break
                hits = result.get("hits", [])
                if not hits:
                    break

                for hit in hits:
                    source = hit.get("_source", {})
                    key = source.get("key")
                    if key:
                        await self._graph._post(
                            "_delete",
                            {
                                "tenant_id": self._graph._config.tenant_id,
                                "user_id": self._graph._config.user_id,
                                "key": key,
                                "namespace": namespace,
                            },
                            allow_404=True,
                        )

                # Use search_after for pagination.
                last_sort = hits[-1].get("sort")
                if last_sort:
                    body["search_after"] = last_sort
                else:
                    break

    async def get_schema_info(self) -> dict[str, Any]:
        """Return metadata about the Memory Store schema."""
        return {
            "backend": "memory_store",
            "auto_managed": True,
            "description": (
                "Indices are auto-created on first write. "
                "Vector mappings are created on first embedding upsert."
            ),
        }
