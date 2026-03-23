"""Neo4j implementation of the GraphBackend protocol.

Translates semantic graph operations (nodes, relationships, traversals)
into Cypher queries executed against a Neo4j database via ``Neo4jClient``.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from neo4j_agent_memory.graph.client import Neo4jClient

# ---------------------------------------------------------------------------
# Label / identifier sanitisation
# ---------------------------------------------------------------------------

_VALID_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _sanitize(name: str, kind: str = "label") -> str:
    """Validate that *name* is a safe Cypher identifier.

    Only letters, digits, and underscores are permitted, and the name
    must start with a letter.  This prevents Cypher injection when names
    are interpolated into query strings (Neo4j does not support
    parameterised labels or relationship types).

    Raises:
        ValueError: If *name* does not match the safe pattern.
    """
    if not name or not _VALID_IDENTIFIER.match(name):
        raise ValueError(
            f"Invalid {kind} name {name!r}. "
            "Must start with a letter and contain only [A-Za-z0-9_]."
        )
    return name


def _sanitize_property(name: str) -> str:
    """Validate a property name for safe use in Cypher expressions."""
    return _sanitize(name, kind="property")


# ---------------------------------------------------------------------------
# Neo4jGraphBackend
# ---------------------------------------------------------------------------


class Neo4jGraphBackend:
    """``GraphBackend`` implementation backed by Neo4j.

    All public methods correspond to the ``GraphBackend`` protocol defined
    in ``backend_protocol.py``.  Cypher queries are built with sanitised
    labels and parameterised property values.
    """

    def __init__(self, client: Neo4jClient) -> None:
        self._client = client

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        """Establish the backend connection."""
        await self._client.connect()

    async def close(self) -> None:
        """Release all backend resources."""
        await self._client.close()

    @property
    def is_connected(self) -> bool:
        """Return ``True`` when the backend is ready for operations."""
        return self._client.is_connected

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _build_where(
        filters: dict[str, Any],
        params: dict[str, Any],
        node_var: str = "n",
        prefix: str = "f",
    ) -> str:
        """Build a WHERE clause from an equality-filter dict.

        Each filter key is validated as a safe property name.  Values are
        passed as query parameters (``$f_<key>``), never interpolated.

        Returns an empty string when *filters* is empty.
        """
        if not filters:
            return ""
        parts: list[str] = []
        for key, value in filters.items():
            safe_key = _sanitize_property(key)
            param_name = f"{prefix}_{safe_key}"
            parts.append(f"{node_var}.{safe_key} = ${param_name}")
            params[param_name] = value
        return " WHERE " + " AND ".join(parts)

    @staticmethod
    def _props_set_clause(
        props: dict[str, Any],
        params: dict[str, Any],
        node_var: str = "n",
        prefix: str = "p",
    ) -> str:
        """Build ``SET n.key = $p_key, ...`` from a properties dict."""
        parts: list[str] = []
        for key, value in props.items():
            safe_key = _sanitize_property(key)
            param_name = f"{prefix}_{safe_key}"
            parts.append(f"{node_var}.{safe_key} = ${param_name}")
            params[param_name] = value
        return ", ".join(parts)

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
        """Create or update a node via MERGE on ``(n:{label} {id: $id})``."""
        safe_label = _sanitize(label)
        params: dict[str, Any] = {"id": id}

        # ON CREATE SET -- all supplied properties
        create_props = {**properties, "id": id}
        create_parts: list[str] = []
        for key, value in create_props.items():
            safe_key = _sanitize_property(key)
            param_name = f"c_{safe_key}"
            create_parts.append(f"n.{safe_key} = ${param_name}")
            params[param_name] = value

        # ON MATCH SET -- explicit overrides, or fall back to all properties
        match_props = on_match_update if on_match_update is not None else properties
        match_parts: list[str] = []
        for key, value in match_props.items():
            safe_key = _sanitize_property(key)
            param_name = f"m_{safe_key}"
            match_parts.append(f"n.{safe_key} = ${param_name}")
            params[param_name] = value

        query = f"MERGE (n:{safe_label} {{id: $id}})\n"
        if create_parts:
            query += "ON CREATE SET " + ", ".join(create_parts) + "\n"
        if match_parts:
            query += "ON MATCH SET " + ", ".join(match_parts) + "\n"

        # Additional labels
        if additional_labels:
            label_fragments = []
            for extra in additional_labels:
                safe_extra = _sanitize(extra)
                label_fragments.append(f"n:{safe_extra}")
            query += "SET " + ", ".join(label_fragments) + "\n"

        query += "RETURN n"

        results = await self._client.execute_write(query, params)
        if results:
            return dict(results[0]["n"])
        return {}

    async def get_node(
        self,
        label: str,
        *,
        id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve a single node by id or filters."""
        safe_label = _sanitize(label)
        params: dict[str, Any] = {}

        if id is not None:
            params["id"] = id
            query = f"MATCH (n:{safe_label} {{id: $id}}) RETURN n LIMIT 1"
        else:
            where = self._build_where(filters or {}, params)
            query = f"MATCH (n:{safe_label}){where} RETURN n LIMIT 1"

        results = await self._client.execute_read(query, params)
        if results:
            return dict(results[0]["n"])
        return None

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
        """Query multiple nodes with optional filtering and pagination."""
        safe_label = _sanitize(label)
        params: dict[str, Any] = {}
        where = self._build_where(filters or {}, params)

        query = f"MATCH (n:{safe_label}){where}"

        if order_by is not None:
            safe_order = _sanitize_property(order_by)
            direction = "DESC" if order_dir == "desc" else "ASC"
            query += f" ORDER BY n.{safe_order} {direction}"

        if offset > 0:
            query += f" SKIP {int(offset)}"

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        query += " RETURN n"

        results = await self._client.execute_read(query, params)
        return [dict(r["n"]) for r in results]

    async def update_node(
        self,
        label: str,
        id: str,
        properties: dict[str, Any],
        *,
        increment: dict[str, int | float] | None = None,
    ) -> dict[str, Any] | None:
        """Update an existing node's properties."""
        safe_label = _sanitize(label)
        params: dict[str, Any] = {"id": id}

        set_parts: list[str] = []

        # Regular property updates
        for key, value in properties.items():
            safe_key = _sanitize_property(key)
            param_name = f"p_{safe_key}"
            set_parts.append(f"n.{safe_key} = ${param_name}")
            params[param_name] = value

        # Atomic increments
        if increment:
            for key, value in increment.items():
                safe_key = _sanitize_property(key)
                param_name = f"inc_{safe_key}"
                set_parts.append(
                    f"n.{safe_key} = COALESCE(n.{safe_key}, 0) + ${param_name}"
                )
                params[param_name] = value

        if not set_parts:
            # Nothing to update -- just return the node as-is
            return await self.get_node(label, id=id)

        query = (
            f"MATCH (n:{safe_label} {{id: $id}})\n"
            f"SET {', '.join(set_parts)}\n"
            "RETURN n"
        )

        results = await self._client.execute_write(query, params)
        if results:
            return dict(results[0]["n"])
        return None

    async def delete_node(
        self,
        label: str,
        id: str,
        *,
        detach: bool = True,
    ) -> bool:
        """Delete a node, optionally detaching relationships first."""
        safe_label = _sanitize(label)
        delete_keyword = "DETACH DELETE" if detach else "DELETE"
        query = (
            f"MATCH (n:{safe_label} {{id: $id}})\n"
            f"{delete_keyword} n\n"
            "RETURN count(n) AS deleted"
        )
        results = await self._client.execute_write(query, {"id": id})
        return results[0]["deleted"] > 0 if results else False

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
        """Create or update a relationship between two nodes."""
        safe_from = _sanitize(from_label)
        safe_to = _sanitize(to_label)
        safe_rel = _sanitize(relationship_type, kind="relationship type")

        params: dict[str, Any] = {"from_id": from_id, "to_id": to_id}

        create_verb = "MERGE" if upsert else "CREATE"

        query = (
            f"MATCH (a:{safe_from} {{id: $from_id}})\n"
            f"MATCH (b:{safe_to} {{id: $to_id}})\n"
            f"{create_verb} (a)-[r:{safe_rel}]->(b)\n"
        )

        if properties:
            set_clause = self._props_set_clause(properties, params, node_var="r", prefix="rp")
            query += f"SET {set_clause}\n"

        query += "RETURN properties(r) AS rel_props"

        results = await self._client.execute_write(query, params)
        if results:
            return dict(results[0]["rel_props"])
        return None

    async def unlink_nodes(
        self,
        from_label: str,
        from_id: str,
        to_label: str,
        to_id: str,
        relationship_type: str,
    ) -> bool:
        """Remove a specific relationship between two nodes."""
        safe_from = _sanitize(from_label)
        safe_to = _sanitize(to_label)
        safe_rel = _sanitize(relationship_type, kind="relationship type")

        query = (
            f"MATCH (a:{safe_from} {{id: $from_id}})"
            f"-[r:{safe_rel}]->"
            f"(b:{safe_to} {{id: $to_id}})\n"
            "DELETE r\n"
            "RETURN count(r) AS deleted"
        )
        params = {"from_id": from_id, "to_id": to_id}
        results = await self._client.execute_write(query, params)
        return results[0]["deleted"] > 0 if results else False

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
        """Traverse relationships from a starting node."""
        safe_start = _sanitize(start_label)
        params: dict[str, Any] = {"start_id": start_id}

        # Relationship type filter
        if relationship_types:
            rel_types = "|".join(_sanitize(rt, kind="relationship type") for rt in relationship_types)
            rel_pattern = f":{rel_types}"
        else:
            rel_pattern = ""

        # Variable-length path with depth
        depth_range = f"*1..{int(depth)}"

        # Direction arrows
        if direction == "outgoing":
            left_arrow, right_arrow = "-", "->"
        elif direction == "incoming":
            left_arrow, right_arrow = "<-", "-"
        else:
            left_arrow, right_arrow = "-", "-"

        # Target label filter
        target_clause = ""
        if target_labels:
            safe_targets = [_sanitize(tl) for tl in target_labels]
            target_clause = ":" + ":".join(safe_targets)

        # Build path query
        query = (
            f"MATCH (start:{safe_start} {{id: $start_id}})\n"
            f"MATCH p = (start){left_arrow}[r{rel_pattern}{depth_range}]{right_arrow}(end{target_clause})\n"
        )

        if include_edges:
            # Return end node properties plus relationship properties of the
            # last relationship in the path.
            query += (
                "WITH end, relationships(p) AS rels\n"
                "RETURN properties(end) AS node, properties(rels[-1]) AS edge"
            )
        else:
            query += "RETURN DISTINCT properties(end) AS node"

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        results = await self._client.execute_read(query, params)

        output: list[dict[str, Any]] = []
        for record in results:
            node_props = dict(record["node"])
            if include_edges and "edge" in record:
                node_props["_edge"] = dict(record["edge"])
            output.append(node_props)
        return output

    # -- aggregation ---------------------------------------------------------

    async def count_nodes(
        self,
        label: str,
        *,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Count nodes matching a label and optional filters."""
        safe_label = _sanitize(label)
        params: dict[str, Any] = {}
        where = self._build_where(filters or {}, params)
        query = f"MATCH (n:{safe_label}){where} RETURN count(n) AS cnt"
        results = await self._client.execute_read(query, params)
        return results[0]["cnt"] if results else 0

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
    ) -> list[dict[str, Any]]:
        """Semantic vector similarity search.

        The index name is derived from the label and property name using the
        convention ``{label_lower}_{property_name}_idx``.  For example,
        label ``"Message"`` with property ``"embedding"`` maps to index
        ``"message_embedding_idx"``.
        """
        # Sanitise inputs (label and property_name are used in index name derivation only)
        _sanitize(label)
        _sanitize_property(property_name)

        index_name = f"{label.lower()}_{property_name}_idx"

        params: dict[str, Any] = {
            "index_name": index_name,
            "embedding": query_embedding,
            "limit": limit,
            "threshold": threshold,
        }

        # Base vector search query
        query = (
            "CALL db.index.vector.queryNodes($index_name, $limit, $embedding)\n"
            "YIELD node, score\n"
            "WHERE score >= $threshold\n"
        )

        # Optional post-search property filters
        if filters:
            filter_parts: list[str] = []
            for key, value in filters.items():
                safe_key = _sanitize_property(key)
                param_name = f"vf_{safe_key}"
                filter_parts.append(f"node.{safe_key} = ${param_name}")
                params[param_name] = value
            query += "AND " + " AND ".join(filter_parts) + "\n"

        query += "RETURN node, score ORDER BY score DESC"

        results = await self._client.execute_read(query, params)

        output: list[dict[str, Any]] = []
        for record in results:
            node_props = dict(record["node"])
            node_props["_score"] = record["score"]
            output.append(node_props)
        return output

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
        """Atomically create a node and its relationships in one Cypher query."""
        safe_label = _sanitize(label)
        params: dict[str, Any] = {"id": id}

        # Build SET clause for the new node's properties (including id)
        all_props = {**properties, "id": id}
        set_parts: list[str] = []
        for key, value in all_props.items():
            safe_key = _sanitize_property(key)
            param_name = f"np_{safe_key}"
            set_parts.append(f"n.{safe_key} = ${param_name}")
            params[param_name] = value

        query = f"CREATE (n:{safe_label})\n"
        if set_parts:
            query += "SET " + ", ".join(set_parts) + "\n"

        # Additional labels
        if additional_labels:
            label_fragments = [f"n:{_sanitize(extra)}" for extra in additional_labels]
            query += "SET " + ", ".join(label_fragments) + "\n"

        # Relationships
        if links:
            with_vars = ["n"]
            for idx, link in enumerate(links):
                target_label = _sanitize(link["target_label"])
                rel_type = _sanitize(link["relationship_type"], kind="relationship type")
                direction = link.get("direction", "outgoing")

                target_id_param = f"link_{idx}_id"
                params[target_id_param] = link["target_id"]

                match_alias = f"t{idx}"
                rel_alias = f"r{idx}"
                with_vars.append(match_alias)

                # Carry forward all previously matched variables
                query += "WITH " + ", ".join(with_vars[:-1]) + "\n"
                query += (
                    f"OPTIONAL MATCH ({match_alias}:{target_label} "
                    f"{{id: ${target_id_param}}})\n"
                )

                # Build FOREACH to conditionally create the relationship
                # only when the target node exists.
                if direction == "incoming":
                    rel_clause = f"({match_alias})-[{rel_alias}:{rel_type}]->(n)"
                else:
                    rel_clause = f"(n)-[{rel_alias}:{rel_type}]->({match_alias})"

                # Set relationship properties if provided
                link_props = link.get("properties")
                if link_props:
                    rp_parts: list[str] = []
                    for pk, pv in link_props.items():
                        safe_pk = _sanitize_property(pk)
                        rp_param = f"lp_{idx}_{safe_pk}"
                        rp_parts.append(f"{rel_alias}.{safe_pk} = ${rp_param}")
                        params[rp_param] = pv
                    set_rp = " SET " + ", ".join(rp_parts)
                else:
                    set_rp = ""

                query += (
                    f"FOREACH (_ IN CASE WHEN {match_alias} IS NOT NULL THEN [1] ELSE [] END |\n"
                    f"  CREATE {rel_clause}{set_rp}\n"
                    ")\n"
                )

        query += "RETURN n"

        results = await self._client.execute_write(query, params)
        if results:
            return dict(results[0]["n"])
        return {}

    # -- TTL / expiration ----------------------------------------------------

    async def expire_node(
        self,
        label: str,
        id: str,
        *,
        ttl_seconds: int,
    ) -> bool:
        """Mark a node for expiration after a TTL.

        Sets ``_expires_at`` to ``datetime() + duration({seconds: $ttl})``.
        """
        safe_label = _sanitize(label)
        query = (
            f"MATCH (n:{safe_label} {{id: $id}})\n"
            "SET n._expires_at = datetime() + duration({seconds: $ttl})\n"
            "RETURN count(n) AS updated"
        )
        params = {"id": id, "ttl": ttl_seconds}
        results = await self._client.execute_write(query, params)
        return results[0]["updated"] > 0 if results else False
