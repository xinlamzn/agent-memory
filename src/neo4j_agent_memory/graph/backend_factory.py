"""Backend factory for assembling a BackendBundle.

The factory inspects ``MemorySettings.backend`` and returns the
appropriate ``BackendBundle`` with all three backend surfaces wired up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neo4j_agent_memory.graph.backend_protocol import (
    BackendBundle,
    BackendCapabilities,
    UnsupportedBackendOperation,
)

if TYPE_CHECKING:
    from neo4j_agent_memory.config.settings import MemorySettings


def create_backend_bundle(settings: "MemorySettings") -> BackendBundle:
    """Create a BackendBundle for the configured backend.

    Args:
        settings: Application settings containing backend selection
            and connection parameters.

    Returns:
        A fully assembled BackendBundle.

    Raises:
        UnsupportedBackendOperation: If a Memory Store backend is
            requested but the configuration is missing.
        ValueError: If the backend name is unknown.
    """
    backend = settings.backend

    if backend == "neo4j":
        return _create_neo4j_bundle(settings)
    elif backend == "memory_store":
        return _create_memory_store_bundle(settings)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Expected 'neo4j' or 'memory_store'.")


def _create_neo4j_bundle(settings: "MemorySettings") -> BackendBundle:
    """Build the Neo4j BackendBundle."""
    from neo4j_agent_memory.graph.client import Neo4jClient
    from neo4j_agent_memory.graph.neo4j_backend import Neo4jGraphBackend
    from neo4j_agent_memory.graph.neo4j_schema_backend import Neo4jSchemaBackend
    from neo4j_agent_memory.graph.neo4j_utility_backend import Neo4jUtilityBackend

    neo4j_client = Neo4jClient(settings.neo4j)

    graph = Neo4jGraphBackend(neo4j_client)
    schema = Neo4jSchemaBackend(
        neo4j_client,
        vector_dimensions=settings.embedding.dimensions,
    )
    utility = Neo4jUtilityBackend(neo4j_client)

    capabilities = BackendCapabilities(
        supports_raw_query=True,
        supports_schema_management=True,
        supports_schema_persistence=True,
        supports_graph_export=True,
        supports_geo_search=True,
        supports_vector_search=True,
        supports_transactions=True,
    )

    return BackendBundle(
        graph=graph,
        schema=schema,
        utility=utility,
        capabilities=capabilities,
        backend_name="neo4j",
        raw=neo4j_client,
    )


def _create_memory_store_bundle(settings: "MemorySettings") -> BackendBundle:
    """Build the Memory Store BackendBundle.

    Raises:
        UnsupportedBackendOperation: If memory_store config is missing.
    """
    if settings.memory_store is None:
        raise UnsupportedBackendOperation(
            "create_backend",
            "memory_store",
            hint="Set settings.memory_store to a MemoryStoreConfig instance.",
        )

    # Stage 2: Memory Store backend implementations will be imported here.
    # For now, raise a clear error indicating this backend is not yet available.
    raise UnsupportedBackendOperation(
        "create_backend",
        "memory_store",
        hint="The memory_store backend is not yet implemented. "
        "It will be available in a future release.",
    )
