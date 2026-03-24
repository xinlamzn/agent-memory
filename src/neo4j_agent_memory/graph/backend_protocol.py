"""Backend-neutral protocol definitions for the agent memory system.

Defines the contract that all backend implementations must fulfill:
- GraphBackend: Semantic graph operations (nodes, relationships, search)
- SchemaBackend: Schema setup and management
- UtilityBackend: Stats, graph export, and location queries
- BackendCapabilities: Feature flags for backend-specific capabilities
- BackendBundle: Assembled bundle of all backends for a given configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class BackendCapabilities:
    """Explicit feature flags for backend capabilities.

    Used to fail clearly when an unsupported feature is requested
    under a backend that cannot provide it.
    """

    supports_raw_query: bool = False
    supports_schema_management: bool = False
    supports_schema_persistence: bool = False
    supports_graph_export: bool = False
    supports_geo_search: bool = False
    supports_vector_search: bool = False
    supports_transactions: bool = False


class UnsupportedBackendOperation(Exception):
    """Raised when an operation is not supported by the current backend."""

    def __init__(self, operation: str, backend: str, *, hint: str | None = None):
        msg = f"Operation '{operation}' is not supported by the '{backend}' backend."
        if hint:
            msg += f" {hint}"
        super().__init__(msg)
        self.operation = operation
        self.backend = backend


# ---------------------------------------------------------------------------
# GraphBackend
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphBackend(Protocol):
    """Semantic graph operations used by memory classes.

    This is the primary interface through which memory classes interact
    with the underlying graph store.  It is intentionally expressed in
    semantic graph terms (nodes, edges, traversals) rather than any
    query language.
    """

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        """Establish the backend connection."""
        ...

    async def close(self) -> None:
        """Release all backend resources."""
        ...

    @property
    def is_connected(self) -> bool:
        """Return True when the backend is ready for operations."""
        ...

    # -- node operations -----------------------------------------------------

    async def upsert_node(
        self,
        label: str,
        *,
        id: str,
        properties: dict[str, Any],
        on_match_update: dict[str, Any] | None = None,
        additional_labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create or update a node.

        Args:
            label: Primary node label (e.g. "Message", "Entity").
            id: Unique node identifier (stored as the ``id`` property).
            properties: Properties to set on create.  Must include all
                required properties for the label.
            on_match_update: Properties to update when the node already
                exists.  If ``None``, updates all supplied properties.
            additional_labels: Extra labels to add (e.g. type/subtype
                labels for Entity nodes).

        Returns:
            The full property map of the node after the upsert.
        """
        ...

    async def get_node(
        self,
        label: str,
        *,
        id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve a single node by id or filters.

        Args:
            label: Node label to match.
            id: Exact node id to look up.
            filters: Property equality filters (used when *id* is ``None``).

        Returns:
            Node property map, or ``None`` if not found.
        """
        ...

    async def query_nodes(
        self,
        label: str,
        *,
        filters: dict[str, Any] | None = None,
        order_by: str | None = None,
        order_dir: Literal["asc", "desc"] = "asc",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Query multiple nodes with optional filtering and pagination.

        Args:
            label: Node label to match.
            filters: Property equality/range filters.
            order_by: Property name to sort on.
            order_dir: Sort direction.
            limit: Maximum nodes to return.
            offset: Number of nodes to skip.

        Returns:
            List of node property maps.
        """
        ...

    async def update_node(
        self,
        label: str,
        id: str,
        properties: dict[str, Any],
        *,
        increment: dict[str, int | float] | None = None,
    ) -> dict[str, Any] | None:
        """Update an existing node's properties.

        Args:
            label: Node label.
            id: Node identifier.
            properties: Properties to overwrite.
            increment: Properties to increment atomically (e.g. counters).

        Returns:
            Updated node property map, or ``None`` if not found.
        """
        ...

    async def delete_node(
        self,
        label: str,
        id: str,
        *,
        detach: bool = True,
    ) -> bool:
        """Delete a node.

        Args:
            label: Node label.
            id: Node identifier.
            detach: If ``True``, delete all attached relationships first.

        Returns:
            ``True`` if the node was deleted.
        """
        ...

    # -- relationship operations ---------------------------------------------

    async def link_nodes(
        self,
        from_label: str,
        from_id: str,
        to_label: str,
        to_id: str,
        relationship_type: str,
        *,
        properties: dict[str, Any] | None = None,
        upsert: bool = True,
    ) -> dict[str, Any] | None:
        """Create or update a relationship between two nodes.

        Args:
            from_label: Source node label.
            from_id: Source node id.
            to_label: Target node label.
            to_id: Target node id.
            relationship_type: Relationship type name.
            properties: Properties on the relationship.
            upsert: If ``True``, merge rather than create.

        Returns:
            Relationship property map (may be empty), or ``None`` if
            either endpoint was not found.
        """
        ...

    async def unlink_nodes(
        self,
        from_label: str,
        from_id: str,
        to_label: str,
        to_id: str,
        relationship_type: str,
    ) -> bool:
        """Remove a specific relationship between two nodes.

        Returns:
            ``True`` if the relationship was removed.
        """
        ...

    # -- traversal -----------------------------------------------------------

    async def traverse(
        self,
        start_label: str,
        start_id: str,
        *,
        relationship_types: list[str] | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "both",
        target_labels: list[str] | None = None,
        depth: int = 1,
        include_edges: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Traverse relationships from a starting node.

        Args:
            start_label: Starting node label.
            start_id: Starting node id.
            relationship_types: Relationship types to follow.
            direction: Traversal direction.
            target_labels: Only return nodes with these labels.
            depth: Maximum traversal depth.
            include_edges: If ``True``, each result dict includes an
                ``_edge`` key with relationship properties.
            limit: Maximum results.

        Returns:
            List of node property maps (optionally with ``_edge``).
        """
        ...

    # -- aggregation ---------------------------------------------------------

    async def count_nodes(
        self,
        label: str,
        *,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Count nodes matching a label and optional filters."""
        ...

    # -- vector search -------------------------------------------------------

    async def vector_search(
        self,
        label: str,
        property_name: str,
        query_embedding: list[float],
        *,
        limit: int = 10,
        threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
        query_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic vector similarity search.

        Args:
            label: Node label (used to identify the vector index).
            property_name: Embedding property name.
            query_embedding: Query embedding vector.
            limit: Maximum results.
            threshold: Minimum similarity score.
            filters: Additional property filters applied post-search.
            query_text: Optional text query for hybrid (vector + BM25) search.
                When provided, backends that support hybrid search will combine
                vector similarity with BM25 text matching for improved recall.
                Backends that don't support hybrid search may ignore this.

        Returns:
            List of dicts, each containing node properties plus a
            ``_score`` key with the similarity score.
        """
        ...

    # -- batch / composite operations ----------------------------------------

    async def create_node_with_links(
        self,
        label: str,
        *,
        id: str,
        properties: dict[str, Any],
        additional_labels: list[str] | None = None,
        links: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Atomically create a node and its relationships.

        This is an optimization entry-point for backends that can perform
        create + link in a single round-trip (e.g. a single Cypher query
        for Neo4j).

        Each item in *links* is a dict with:
            - ``target_label``: str
            - ``target_id``: str
            - ``relationship_type``: str
            - ``properties``: dict (optional)
            - ``direction``: "outgoing" | "incoming" (default "outgoing")

        Args:
            label: Node label.
            id: Node identifier.
            properties: Node properties.
            additional_labels: Extra labels to apply.
            links: Relationships to create.

        Returns:
            Created node property map.
        """
        ...

    # -- TTL / expiration ----------------------------------------------------

    async def expire_node(
        self,
        label: str,
        id: str,
        *,
        ttl_seconds: int,
    ) -> bool:
        """Mark a node for expiration after a TTL.

        Returns:
            ``True`` if the node was found and updated.
        """
        ...


# ---------------------------------------------------------------------------
# SchemaBackend
# ---------------------------------------------------------------------------


@runtime_checkable
class SchemaBackend(Protocol):
    """Backend-specific schema setup and management."""

    async def setup_all(self) -> None:
        """Set up all schema elements (indexes, constraints, etc.)."""
        ...

    async def setup_constraints(self) -> None:
        """Create uniqueness / existence constraints."""
        ...

    async def setup_indexes(self) -> None:
        """Create regular (non-vector) indexes."""
        ...

    async def setup_vector_indexes(self) -> None:
        """Create vector indexes for semantic search."""
        ...

    async def setup_point_indexes(self) -> None:
        """Create geospatial point indexes."""
        ...

    async def drop_all(self) -> None:
        """Drop all memory-related schema elements."""
        ...

    async def get_schema_info(self) -> dict[str, Any]:
        """Return information about the current schema state."""
        ...


# ---------------------------------------------------------------------------
# UtilityBackend
# ---------------------------------------------------------------------------


@runtime_checkable
class UtilityBackend(Protocol):
    """Backend-neutral utility, query, and export surface."""

    async def get_stats(self) -> dict[str, Any]:
        """Return memory statistics (counts per memory type)."""
        ...

    async def get_graph(
        self,
        *,
        memory_types: list[str] | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        include_embeddings: bool = False,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Export memory graph for visualization.

        Returns a dict with ``nodes``, ``relationships``, and ``metadata``.
        """
        ...

    async def get_locations(
        self,
        *,
        session_id: str | None = None,
        has_coordinates: bool = True,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Retrieve Location entities with optional conversation filtering."""
        ...


# ---------------------------------------------------------------------------
# BackendBundle
# ---------------------------------------------------------------------------


@dataclass
class BackendBundle:
    """Assembled bundle of all backend surfaces for a configuration.

    Created by the backend factory and consumed by ``MemoryClient``.
    """

    graph: GraphBackend
    schema: SchemaBackend
    utility: UtilityBackend
    capabilities: BackendCapabilities
    backend_name: str  # "neo4j" or "memory_store"

    # Optional raw backend handles (backend-specific, not part of the
    # neutral API).  These are exposed for advanced use only.
    raw: Any = field(default=None, repr=False)
