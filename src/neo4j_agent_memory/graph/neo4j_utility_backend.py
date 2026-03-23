"""Neo4j implementation of the UtilityBackend protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from neo4j_agent_memory.graph.client import Neo4jClient
from neo4j_agent_memory.graph.queries import GET_MEMORY_STATS


class Neo4jUtilityBackend:
    """Neo4j-backed implementation of :class:`UtilityBackend`.

    Extracts the utility-method logic that previously lived in
    ``MemoryClient`` (``get_stats``, ``get_graph``, ``get_locations``)
    into a standalone class satisfying the ``UtilityBackend`` protocol.
    """

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # get_stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict[str, Any]:
        """Return memory statistics (counts per memory type)."""
        results = await self._client.execute_read(GET_MEMORY_STATS)
        if results:
            return results[0]
        return {
            "conversations": 0,
            "messages": 0,
            "entities": 0,
            "preferences": 0,
            "facts": 0,
            "traces": 0,
        }

    # ------------------------------------------------------------------
    # get_graph
    # ------------------------------------------------------------------

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

        Returns a dict with ``nodes``, ``relationships``, and ``metadata``
        keys.  Nodes and relationships are plain dicts (not Pydantic models).
        """
        if memory_types is None:
            memory_types = ["short_term", "long_term", "reasoning"]

        all_nodes: list[dict[str, Any]] = []
        all_relationships: list[dict[str, Any]] = []
        node_ids_seen: set[str] = set()

        params: dict[str, Any] = {
            "session_id": session_id,
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
            "include_embeddings": include_embeddings,
            "limit": limit,
        }

        # -- short-term memory graph ------------------------------------
        if "short_term" in memory_types:
            try:
                results = await self._client.execute_read(
                    """
                    MATCH (c:Conversation)-[r:HAS_MESSAGE]->(m:Message)
                    WHERE ($session_id IS NULL OR c.session_id = $session_id)
                    WITH c, r, m
                    LIMIT $limit
                    RETURN c, r, m
                    """,
                    params,
                )
                for row in results:
                    conv = dict(row["c"])
                    msg = dict(row["m"])

                    # Add conversation node
                    if conv.get("id") and conv["id"] not in node_ids_seen:
                        props = {k: v for k, v in conv.items() if v is not None}
                        all_nodes.append(
                            {
                                "id": conv["id"],
                                "labels": ["Conversation"],
                                "properties": props,
                            }
                        )
                        node_ids_seen.add(conv["id"])

                    # Add message node
                    if msg.get("id") and msg["id"] not in node_ids_seen:
                        props = {k: v for k, v in msg.items() if v is not None}
                        if not include_embeddings:
                            props.pop("embedding", None)
                        all_nodes.append(
                            {
                                "id": msg["id"],
                                "labels": ["Message"],
                                "properties": props,
                            }
                        )
                        node_ids_seen.add(msg["id"])

                    # Add relationship
                    if conv.get("id") and msg.get("id"):
                        all_relationships.append(
                            {
                                "id": f"{conv['id']}->{msg['id']}",
                                "type": "HAS_MESSAGE",
                                "from_node": conv["id"],
                                "to_node": msg["id"],
                                "properties": {},
                            }
                        )
            except Exception:
                pass  # Skip if query fails

        # -- long-term memory graph -------------------------------------
        if "long_term" in memory_types:
            try:
                results = await self._client.execute_read(
                    """
                    MATCH (e:Entity)
                    WITH e LIMIT $limit
                    OPTIONAL MATCH (e)-[r:RELATED_TO]-(e2:Entity)
                    RETURN e, r, e2
                    """,
                    {"limit": limit},
                )
                for row in results:
                    entity = dict(row["e"])

                    if entity.get("id") and entity["id"] not in node_ids_seen:
                        props = {k: v for k, v in entity.items() if v is not None}
                        if not include_embeddings:
                            props.pop("embedding", None)
                        all_nodes.append(
                            {
                                "id": entity["id"],
                                "labels": ["Entity"],
                                "properties": props,
                            }
                        )
                        node_ids_seen.add(entity["id"])

                    if row.get("r") and row.get("e2"):
                        e2 = dict(row["e2"])
                        if e2.get("id") and e2["id"] not in node_ids_seen:
                            props = {k: v for k, v in e2.items() if v is not None}
                            if not include_embeddings:
                                props.pop("embedding", None)
                            all_nodes.append(
                                {
                                    "id": e2["id"],
                                    "labels": ["Entity"],
                                    "properties": props,
                                }
                            )
                            node_ids_seen.add(e2["id"])

                        rel = dict(row["r"])
                        all_relationships.append(
                            {
                                "id": f"{entity['id']}->{e2['id']}",
                                "type": rel.get("type", "RELATED_TO"),
                                "from_node": entity["id"],
                                "to_node": e2["id"],
                                "properties": {
                                    k: v
                                    for k, v in rel.items()
                                    if k != "type" and v is not None
                                },
                            }
                        )
            except Exception:
                pass

        # -- reasoning memory graph -------------------------------------
        if "reasoning" in memory_types:
            try:
                results = await self._client.execute_read(
                    """
                    MATCH (rt:ReasoningTrace)
                    WHERE ($session_id IS NULL OR rt.session_id = $session_id)
                    WITH rt LIMIT $limit
                    OPTIONAL MATCH (rt)-[r1:HAS_STEP]->(rs:ReasoningStep)
                    OPTIONAL MATCH (rs)-[r2:USES_TOOL]->(tc:ToolCall)
                    RETURN rt, r1, rs, r2, tc
                    """,
                    params,
                )
                for row in results:
                    trace = dict(row["rt"])

                    if trace.get("id") and trace["id"] not in node_ids_seen:
                        props = {k: v for k, v in trace.items() if v is not None}
                        if not include_embeddings:
                            props.pop("task_embedding", None)
                        all_nodes.append(
                            {
                                "id": trace["id"],
                                "labels": ["ReasoningTrace"],
                                "properties": props,
                            }
                        )
                        node_ids_seen.add(trace["id"])

                    if row.get("rs"):
                        step = dict(row["rs"])
                        if step.get("id") and step["id"] not in node_ids_seen:
                            props = {k: v for k, v in step.items() if v is not None}
                            if not include_embeddings:
                                props.pop("embedding", None)
                            all_nodes.append(
                                {
                                    "id": step["id"],
                                    "labels": ["ReasoningStep"],
                                    "properties": props,
                                }
                            )
                            node_ids_seen.add(step["id"])

                        if trace.get("id") and step.get("id"):
                            all_relationships.append(
                                {
                                    "id": f"{trace['id']}->{step['id']}",
                                    "type": "HAS_STEP",
                                    "from_node": trace["id"],
                                    "to_node": step["id"],
                                    "properties": {},
                                }
                            )

                    if row.get("tc") and row.get("rs"):
                        tc = dict(row["tc"])
                        step = dict(row["rs"])
                        if tc.get("id") and tc["id"] not in node_ids_seen:
                            props = {k: v for k, v in tc.items() if v is not None}
                            all_nodes.append(
                                {
                                    "id": tc["id"],
                                    "labels": ["ToolCall"],
                                    "properties": props,
                                }
                            )
                            node_ids_seen.add(tc["id"])

                        if step.get("id") and tc.get("id"):
                            all_relationships.append(
                                {
                                    "id": f"{step['id']}->{tc['id']}",
                                    "type": "USES_TOOL",
                                    "from_node": step["id"],
                                    "to_node": tc["id"],
                                    "properties": {},
                                }
                            )
            except Exception:
                pass

        return {
            "nodes": all_nodes,
            "relationships": all_relationships,
            "metadata": {
                "memory_types": memory_types,
                "session_id": session_id,
                "since": since.isoformat() if since else None,
                "until": until.isoformat() if until else None,
                "include_embeddings": include_embeddings,
                "node_count": len(all_nodes),
                "relationship_count": len(all_relationships),
            },
        }

    # ------------------------------------------------------------------
    # get_locations
    # ------------------------------------------------------------------

    async def get_locations(
        self,
        *,
        session_id: str | None = None,
        has_coordinates: bool = True,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Retrieve Location entities with optional conversation filtering."""
        # Build the query based on whether session_id filtering is needed
        if session_id:
            # Filter to locations mentioned in the specific conversation
            # EXTRACTED_FROM direction: (Entity)-[:EXTRACTED_FROM]->(Message)
            query = """
                MATCH (e:Entity {type: 'LOCATION'})-[:EXTRACTED_FROM]->(m:Message)<-[:HAS_MESSAGE]-(c:Conversation {session_id: $session_id})
                WITH DISTINCT e
                WHERE $has_coordinates = false OR (e.location.latitude IS NOT NULL AND e.location.longitude IS NOT NULL)
                WITH e LIMIT $limit
                OPTIONAL MATCH (e)-[:EXTRACTED_FROM]->(m2:Message)<-[:HAS_MESSAGE]-(c2:Conversation)
                WITH e, collect(DISTINCT {id: c2.id, title: c2.title, session_id: c2.session_id}) as conversations
                RETURN e.id as id,
                       e.name as name,
                       e.subtype as subtype,
                       e.description as description,
                       e.enriched_description as enriched_description,
                       e.wikipedia_url as wikipedia_url,
                       e.location.latitude as latitude,
                       e.location.longitude as longitude,
                       conversations
            """
        else:
            # Return all locations (no session filtering)
            query = """
                MATCH (e:Entity {type: 'LOCATION'})
                WHERE $has_coordinates = false OR (e.location.latitude IS NOT NULL AND e.location.longitude IS NOT NULL)
                WITH e LIMIT $limit
                OPTIONAL MATCH (e)-[:EXTRACTED_FROM]->(m:Message)<-[:HAS_MESSAGE]-(c:Conversation)
                WITH e, collect(DISTINCT {id: c.id, title: c.title, session_id: c.session_id}) as conversations
                RETURN e.id as id,
                       e.name as name,
                       e.subtype as subtype,
                       e.description as description,
                       e.enriched_description as enriched_description,
                       e.wikipedia_url as wikipedia_url,
                       e.location.latitude as latitude,
                       e.location.longitude as longitude,
                       conversations
            """

        params: dict[str, Any] = {
            "session_id": session_id,
            "has_coordinates": has_coordinates,
            "limit": limit,
        }

        try:
            results = await self._client.execute_read(query, params)
            locations = []
            for row in results:
                # Filter out null conversation entries
                convs = [c for c in (row.get("conversations") or []) if c.get("id")]
                locations.append(
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "subtype": row.get("subtype"),
                        "description": row.get("description"),
                        "enriched_description": row.get("enriched_description"),
                        "wikipedia_url": row.get("wikipedia_url"),
                        "latitude": row.get("latitude"),
                        "longitude": row.get("longitude"),
                        "conversations": convs,
                    }
                )
            return locations
        except Exception:
            return []
