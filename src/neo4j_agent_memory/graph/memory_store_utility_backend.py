"""Memory Store implementation of the UtilityBackend protocol.

Provides operational statistics, graph visualization data, and location
queries using the Memory Store REST API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neo4j_agent_memory.graph.memory_store_backend import MemoryStoreGraphBackend

logger = logging.getLogger(__name__)


class MemoryStoreUtilityBackend:
    """``UtilityBackend`` implementation for the Memory Store."""

    def __init__(self, graph: MemoryStoreGraphBackend) -> None:
        self._graph = graph

    async def get_stats(self) -> dict[str, Any]:
        """Collect node counts grouped by namespace and labels.

        Calls ``_count`` with ``group_by="namespace"`` and
        ``group_by="labels"`` to produce aggregate statistics.
        """
        base = {
            "tenant_id": self._graph._config.tenant_id,
            "user_id": self._graph._config.user_id,
        }

        # Count by namespace
        ns_body = {**base, "group_by": "namespace"}
        ns_result = await self._graph._post("_count", ns_body)

        # Count by labels
        label_body = {**base, "group_by": "labels"}
        label_result = await self._graph._post("_count", label_body)

        return {
            "backend": "memory_store",
            "by_namespace": ns_result if ns_result else {},
            "by_labels": label_result if label_result else {},
        }

    async def get_graph(
        self,
        memory_types: list[str] | None = None,
        session_id: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Assemble a graph visualization payload.

        Uses ``_query`` enumerate to gather nodes, then ``_traverse``
        with ``include_edges=True`` from sampled nodes to discover
        relationships.
        """
        from neo4j_agent_memory.graph.memory_store_backend import LABEL_NAMESPACE

        all_nodes: list[dict[str, Any]] = []
        all_edges: list[dict[str, Any]] = []

        # Determine which labels to query.
        if memory_types:
            labels_to_query = [
                lbl for lbl, ns in LABEL_NAMESPACE.items()
                if ns in memory_types or lbl in memory_types
            ]
        else:
            labels_to_query = list(LABEL_NAMESPACE.keys())

        nodes_remaining = limit
        for label in labels_to_query:
            if nodes_remaining <= 0:
                break

            body: dict[str, Any] = {
                "tenant_id": self._graph._config.tenant_id,
                "user_id": self._graph._config.user_id,
                "labels": [label],
                "top_k": min(nodes_remaining, 100),
            }
            ns = LABEL_NAMESPACE.get(label)
            if ns:
                body["namespace"] = ns
            if session_id:
                body["session_id"] = session_id

            result = await self._graph._post("_query", body)
            if not result:
                continue

            for hit in result.get("hits", []):
                source = hit.get("_source", {})
                node_key = source.get("key", "")
                node_labels = source.get("labels", [label])
                node_props = source.get("properties", {})
                all_nodes.append({
                    "id": node_key,
                    "labels": node_labels,
                    "properties": node_props,
                })
                nodes_remaining -= 1

        # Sample up to 20 nodes for relationship discovery via traverse.
        sample_nodes = all_nodes[:20]
        for node in sample_nodes:
            node_id = node["id"]
            node_label = node["labels"][0] if node["labels"] else "Entity"
            try:
                traversal = await self._graph.traverse(
                    start_label=node_label,
                    start_id=node_id,
                    direction="both",
                    depth=1,
                    include_edges=True,
                    limit=10,
                )
                for t_node in traversal:
                    edge = t_node.pop("_edge", None)
                    if edge:
                        all_edges.append({
                            "from": node_id,
                            "to": t_node.get("id", ""),
                            "type": edge.get("type", "RELATED_TO"),
                            "properties": {
                                k: v for k, v in edge.items() if k != "type"
                            },
                        })
            except Exception as e:
                logger.debug(f"Traverse failed for {node_id}: {e}")

        return {
            "nodes": all_nodes,
            "relationships": all_edges,
            "metadata": {
                "node_count": len(all_nodes),
                "relationship_count": len(all_edges),
                "memory_types": memory_types,
                "session_id": session_id,
            },
        }

    async def get_locations(
        self,
        session_id: str | None = None,
        has_coordinates: bool = True,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query Location entities with optional coordinate filtering.

        Uses ``_query`` with ``labels=["Entity"]`` and filters for
        ``entity_type="LOCATION"`` client-side, then optionally filters
        for presence of ``latitude``/``longitude``.
        """
        body: dict[str, Any] = {
            "tenant_id": self._graph._config.tenant_id,
            "user_id": self._graph._config.user_id,
            "namespace": "long_term",
            "labels": ["Entity"],
            "top_k": min(limit * 5, 10000),  # overfetch for client-side filtering
        }
        if session_id:
            body["session_id"] = session_id

        result = await self._graph._post("_query", body)
        if not result:
            return []

        locations: list[dict[str, Any]] = []
        for hit in result.get("hits", []):
            source = hit.get("_source", {})
            props = source.get("properties", {})

            # Filter for LOCATION entity type.
            if props.get("entity_type") != "LOCATION" and props.get("type") != "LOCATION":
                continue

            # Filter for coordinate presence if requested.
            if has_coordinates:
                if "latitude" not in props or "longitude" not in props:
                    continue

            locations.append({
                "id": source.get("key", ""),
                "name": props.get("displayName", props.get("name", "")),
                "latitude": props.get("latitude"),
                "longitude": props.get("longitude"),
                "properties": props,
            })

            if len(locations) >= limit:
                break

        return locations
