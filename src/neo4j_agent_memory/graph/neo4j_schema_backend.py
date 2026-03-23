"""Neo4j implementation of the SchemaBackend protocol.

Thin wrapper around :class:`SchemaManager` so that ``MemoryClient`` can
depend on the backend-neutral :class:`SchemaBackend` protocol instead of
the concrete Neo4j schema manager.
"""

from __future__ import annotations

from typing import Any

from neo4j_agent_memory.graph.client import Neo4jClient
from neo4j_agent_memory.graph.schema import SchemaManager


class Neo4jSchemaBackend:
    """SchemaBackend implementation backed by Neo4j.

    Every public method required by
    :class:`~neo4j_agent_memory.graph.backend_protocol.SchemaBackend`
    delegates directly to an internal :class:`SchemaManager` instance.
    """

    def __init__(
        self,
        client: Neo4jClient,
        *,
        vector_dimensions: int = 1536,
    ) -> None:
        """Initialize the Neo4j schema backend.

        Args:
            client: An active Neo4j client connection.
            vector_dimensions: Dimensionality for vector indexes
                (default ``1536``).
        """
        self._schema_manager = SchemaManager(
            client,
            vector_dimensions=vector_dimensions,
        )

    # -- SchemaBackend protocol methods -------------------------------------

    async def setup_all(self) -> None:
        """Set up all schema elements (indexes, constraints, etc.)."""
        await self._schema_manager.setup_all()

    async def setup_constraints(self) -> None:
        """Create uniqueness / existence constraints."""
        await self._schema_manager.setup_constraints()

    async def setup_indexes(self) -> None:
        """Create regular (non-vector) indexes."""
        await self._schema_manager.setup_indexes()

    async def setup_vector_indexes(self) -> None:
        """Create vector indexes for semantic search."""
        await self._schema_manager.setup_vector_indexes()

    async def setup_point_indexes(self) -> None:
        """Create geospatial point indexes."""
        await self._schema_manager.setup_point_indexes()

    async def drop_all(self) -> None:
        """Drop all memory-related schema elements."""
        await self._schema_manager.drop_all()

    async def get_schema_info(self) -> dict[str, Any]:
        """Return information about the current schema state."""
        return await self._schema_manager.get_schema_info()
