"""Memory Store implementation of the GraphBackend protocol.

Translates semantic graph operations (nodes, relationships, traversals)
into REST API calls to the OpenSearch Graph Plugin's memory store.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

import aiohttp

from neo4j_agent_memory.config.memory_store_settings import MemoryStoreConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label-to-namespace mapping
# ---------------------------------------------------------------------------

LABEL_NAMESPACE: dict[str, str] = {
    "Conversation": "short_term",
    "Message": "short_term",
    "Entity": "long_term",
    "Preference": "long_term",
    "Fact": "long_term",
    "ReasoningTrace": "reasoning",
    "ReasoningStep": "reasoning",
    "Step": "reasoning",
    "Tool": "reasoning",
    "ToolCall": "reasoning",
}

# Default edge types used when relationship_types is None (the Memory Store
# _traverse endpoint requires at least one edge type).
_DEFAULT_EDGE_TYPES = [
    "RELATED_TO",
    "EXTRACTED_FROM",
    "HAS_MESSAGE",
    "HAS_STEP",
    "USES_TOOL",
    "MENTIONS",
    "DERIVED_FROM",
]

# Protocol direction values → Memory Store API direction values.
_DIRECTION_MAP: dict[str, str] = {
    "outgoing": "out",
    "incoming": "in",
    "both": "both",
}


# ---------------------------------------------------------------------------
# MemoryStoreGraphBackend
# ---------------------------------------------------------------------------


class MemoryStoreGraphBackend:
    """``GraphBackend`` implementation backed by the Memory Store REST API.

    All public methods correspond to the ``GraphBackend`` protocol defined
    in ``backend_protocol.py``.  Requests are issued via ``aiohttp`` to
    the ``_plugins/_graph/memory/{database}/`` REST endpoints.
    """

    def __init__(self, config: MemoryStoreConfig) -> None:
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        self._base_url: str = ""
        self._ssl: bool | None = None

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        """Establish the backend connection."""
        if self._session is not None and not self._session.closed:
            return

        # Build auth
        auth = None
        headers: dict[str, str] = {}
        if self._config.auth_token:
            headers["Authorization"] = (
                f"Bearer {self._config.auth_token.get_secret_value()}"
            )
        elif self._config.username and self._config.password:
            auth = aiohttp.BasicAuth(
                self._config.username,
                self._config.password.get_secret_value(),
            )

        timeout = aiohttp.ClientTimeout(
            connect=self._config.connect_timeout,
            total=self._config.read_timeout,
        )

        self._ssl = None if self._config.verify_ssl else False

        self._session = aiohttp.ClientSession(
            auth=auth,
            headers=headers,
            timeout=timeout,
        )
        self._base_url = (
            f"{self._config.endpoint.rstrip('/')}/"
            f"_plugins/_graph/memory/{self._config.database}"
        )

        # Verify connectivity with a lightweight count request.
        try:
            await self._post("_count", {
                "tenant_id": self._config.tenant_id,
                "user_id": self._config.user_id,
            })
        except Exception as e:
            await self.close()
            from neo4j_agent_memory.core.exceptions import ConnectionError as ConnErr

            raise ConnErr(f"Failed to connect to Memory Store: {e}") from e

    async def close(self) -> None:
        """Release all backend resources."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def is_connected(self) -> bool:
        """Return ``True`` when the backend is ready for operations."""
        return self._session is not None and not self._session.closed

    # -- central POST helper -------------------------------------------------

    async def _post(
        self,
        endpoint: str,
        body: dict[str, Any],
        *,
        allow_404: bool = False,
        max_retries: int = 3,
    ) -> dict[str, Any] | None:
        """POST to a memory store endpoint with retry + exponential backoff.

        Retries on 429 (Too Many Requests) and 5xx errors.

        Args:
            endpoint: Path suffix after the base URL (e.g. ``"_upsert"``).
            body: JSON request body.
            allow_404: When ``True``, return ``None`` on 404 instead of
                raising.
            max_retries: Maximum number of retry attempts.

        Returns:
            Parsed JSON response dict, or ``None`` when *allow_404* is set
            and the server returned 404.

        Raises:
            ConnectionError: On auth failures or connectivity issues.
            ValueError: On 400 Bad Request.
            RuntimeError: On other HTTP errors after retries exhausted.
        """
        if self._session is None or self._session.closed:
            from neo4j_agent_memory.core.exceptions import ConnectionError as ConnErr

            raise ConnErr(
                "Not connected to Memory Store. Call connect() first."
            )

        url = f"{self._base_url}/{endpoint}"
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                async with self._session.post(
                    url, json=body, ssl=self._ssl
                ) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        if attempt < max_retries:
                            delay = min(0.1 * (2 ** attempt), 5.0)
                            await asyncio.sleep(delay)
                            continue
                        text = await resp.text()
                        raise RuntimeError(
                            f"Memory Store returned {resp.status} after "
                            f"{max_retries} retries: {text}"
                        )

                    if resp.status == 404 and allow_404:
                        return None

                    if resp.status == 400:
                        text = await resp.text()
                        raise ValueError(
                            f"Memory Store bad request: {text}"
                        )

                    if resp.status in (401, 403):
                        text = await resp.text()
                        from neo4j_agent_memory.core.exceptions import (
                            ConnectionError as ConnErr,
                        )

                        raise ConnErr(
                            f"Memory Store auth error ({resp.status}): {text}"
                        )

                    if resp.status >= 400:
                        text = await resp.text()
                        raise RuntimeError(
                            f"Memory Store error ({resp.status}): {text}"
                        )

                    return await resp.json()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < max_retries:
                    delay = min(0.1 * (2 ** attempt), 5.0)
                    await asyncio.sleep(delay)
                    continue
                from neo4j_agent_memory.core.exceptions import (
                    ConnectionError as ConnErr,
                )

                raise ConnErr(
                    f"Memory Store connection failed: {e}"
                ) from e

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unexpected retry exhaustion")

    # -- scope / parsing helpers ---------------------------------------------

    def _scope(self, label: str | None = None) -> dict[str, Any]:
        """Return base scope fields for a request body."""
        d: dict[str, Any] = {
            "tenant_id": self._config.tenant_id,
            "user_id": self._config.user_id,
        }
        if label:
            ns = LABEL_NAMESPACE.get(label)
            if ns:
                d["namespace"] = ns
        return d

    @staticmethod
    def _parse_query_hit(hit: dict[str, Any]) -> dict[str, Any]:
        """Parse a ``_query`` response hit into a node property dict.

        The returned dict mirrors what the Neo4j backend returns: a flat
        property map with ``id`` set to the node key and an optional
        ``_score`` field.
        """
        source = hit.get("_source", {})
        props = dict(source.get("properties", {}))
        props["id"] = source.get("key", "")
        labels = source.get("labels")
        if labels:
            props["_labels"] = labels
        score = hit.get("_score")
        if score is not None:
            props["_score"] = score
        return props

    @staticmethod
    def _parse_traverse_result(
        result: dict[str, Any],
        include_edges: bool,
    ) -> list[dict[str, Any]]:
        """Parse a ``_traverse`` response into a list of node dicts."""
        output: list[dict[str, Any]] = []
        for hit in result.get("hits", []):
            # Traverse response may nest node data under "node" key
            if "node" in hit:
                node_source = hit["node"].get("_source", {})
            else:
                node_source = hit.get("_source", {})
            props = dict(node_source.get("properties", {}))
            props["id"] = node_source.get("key", "")
            labels = node_source.get("labels")
            if labels:
                props["_labels"] = labels
            if include_edges and "edge" in hit:
                edge_data = hit["edge"]
                edge_props: dict[str, Any] = {}
                if isinstance(edge_data, dict):
                    edge_source = edge_data.get("_source", edge_data)
                    edge_props = dict(edge_source.get("properties", {}))
                    if "type" in edge_source:
                        edge_props["type"] = edge_source["type"]
                props["_edge"] = edge_props
            output.append(props)
        return output

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
        """Create or update a node via ``_upsert``."""
        body: dict[str, Any] = {
            **self._scope(label),
            "key": id,
            "labels": [label] + (additional_labels or []),
            "properties": {**properties, "id": id},
        }
        # Promote embedding to top-level for kNN indexing.
        for embed_key in ("embedding", "task_embedding"):
            if embed_key in properties and isinstance(properties[embed_key], list):
                body["embedding"] = properties[embed_key]
                break

        await self._post("_upsert", body)
        return {**properties, "id": id}

    async def get_node(
        self,
        label: str,
        *,
        id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve a single node by id or filters via ``_query`` enumerate."""
        body: dict[str, Any] = {
            **self._scope(label),
            "labels": [label],
            "top_k": 100,
        }
        # Use BM25 text search to narrow results when looking up by id.
        if id is not None:
            body["query_text"] = id

        result = await self._post("_query", body)
        if not result:
            return None

        for hit in result.get("hits", []):
            parsed = self._parse_query_hit(hit)
            if id is not None:
                if parsed.get("id") == id:
                    parsed.pop("_score", None)
                    return parsed
            elif filters:
                if all(parsed.get(k) == v for k, v in filters.items()):
                    parsed.pop("_score", None)
                    return parsed
            else:
                parsed.pop("_score", None)
                return parsed

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
        """Query multiple nodes with optional filtering via ``_query`` enumerate."""
        body: dict[str, Any] = {
            **self._scope(label),
            "labels": [label],
            "top_k": limit or 100,
        }

        result = await self._post("_query", body)
        if not result:
            return []

        hits = result.get("hits", [])
        parsed = [self._parse_query_hit(h) for h in hits]

        # Apply property filters client-side (the query API does not support
        # arbitrary property filters).
        if filters:
            parsed = [
                p for p in parsed
                if all(p.get(k) == v for k, v in filters.items())
            ]

        # Apply offset client-side.
        if offset > 0:
            parsed = parsed[offset:]

        # Strip _score from enumerate results.
        for p in parsed:
            p.pop("_score", None)

        return parsed

    async def update_node(
        self,
        label: str,
        id: str,
        properties: dict[str, Any],
        *,
        increment: dict[str, int | float] | None = None,
    ) -> dict[str, Any] | None:
        """Update an existing node's properties via ``_update``."""
        body: dict[str, Any] = {
            **self._scope(label),
            "key": id,
        }
        if properties:
            body["set"] = properties
        if increment:
            body["increment"] = increment

        # Promote embedding updates to the dedicated field.
        for embed_key in ("embedding", "task_embedding"):
            if embed_key in properties and isinstance(properties[embed_key], list):
                body["set_embedding"] = properties[embed_key]
                body["set_embedding_field"] = embed_key
                break

        result = await self._post("_update", body, allow_404=True)
        if result is None:
            return None
        return {**properties, "id": id}

    async def delete_node(
        self,
        label: str,
        id: str,
        *,
        detach: bool = True,
    ) -> bool:
        """Delete a node via ``_delete``."""
        body: dict[str, Any] = {
            **self._scope(label),
            "key": id,
            "cascade": detach,
        }
        result = await self._post("_delete", body, allow_404=True)
        if result is None:
            return False
        return result.get("deleted", False)

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
        """Create or update a relationship via ``_link``."""
        body: dict[str, Any] = {
            **self._scope(from_label),
            "source_key": from_id,
            "target_key": to_id,
            "relation_type": relationship_type,
        }
        if properties:
            body["properties"] = properties

        result = await self._post("_link", body, allow_404=True)
        if result is None:
            return None
        return properties or {}

    async def unlink_nodes(
        self,
        from_label: str,
        from_id: str,
        to_label: str,
        to_id: str,
        relationship_type: str,
    ) -> bool:
        """Remove a specific relationship via ``_unlink``."""
        body: dict[str, Any] = {
            **self._scope(from_label),
            "source_key": from_id,
            "target_key": to_id,
            "relation_type": relationship_type,
        }
        result = await self._post("_unlink", body, allow_404=True)
        if result is None:
            return False
        return result.get("unlinked", False)

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
        """Traverse relationships from a starting node via ``_traverse``."""
        edge_types = relationship_types or _DEFAULT_EDGE_TYPES
        api_direction = _DIRECTION_MAP.get(direction, "both")

        body: dict[str, Any] = {
            **self._scope(start_label),
            "start_key": start_id,
            "edge_types": edge_types,
            "direction": api_direction,
            "max_depth": depth,
            "include_edges": include_edges,
        }
        if target_labels:
            body["target_labels"] = target_labels
        if limit is not None:
            body["limit"] = limit

        result = await self._post("_traverse", body, allow_404=True)
        if not result:
            return []

        return self._parse_traverse_result(result, include_edges)

    # -- aggregation ---------------------------------------------------------

    async def count_nodes(
        self,
        label: str,
        *,
        filters: dict[str, Any] | None = None,
    ) -> int:
        """Count nodes matching a label via ``_count``."""
        body: dict[str, Any] = {
            **self._scope(label),
            "labels": [label],
        }
        if filters:
            body["filters"] = filters

        result = await self._post("_count", body)
        if not result:
            return 0
        return result.get("count", 0)

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
        """Semantic vector similarity search via ``_query`` vector mode."""
        body: dict[str, Any] = {
            **self._scope(label),
            "query_vector": query_embedding,
            "labels": [label],
            "top_k": limit,
        }
        if property_name != "embedding":
            body["vector_field"] = property_name

        result = await self._post("_query", body)
        if not result:
            return []

        output: list[dict[str, Any]] = []
        for hit in result.get("hits", []):
            parsed = self._parse_query_hit(hit)
            score = parsed.get("_score", 0.0)
            if score >= threshold:
                output.append(parsed)
            # Apply property filters client-side if needed.
            if filters and not all(
                parsed.get(k) == v for k, v in filters.items()
            ):
                if output and output[-1] is parsed:
                    output.pop()

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
        """Create a node and its relationships (upsert + N x link)."""
        node = await self.upsert_node(
            label,
            id=id,
            properties=properties,
            additional_labels=additional_labels,
        )

        if links:
            for link in links:
                target_label = link["target_label"]
                target_id = link["target_id"]
                rel_type = link["relationship_type"]
                link_props = link.get("properties")
                link_direction = link.get("direction", "outgoing")

                if link_direction == "incoming":
                    await self.link_nodes(
                        target_label, target_id,
                        label, id,
                        rel_type,
                        properties=link_props,
                    )
                else:
                    await self.link_nodes(
                        label, id,
                        target_label, target_id,
                        rel_type,
                        properties=link_props,
                    )

        return node

    # -- TTL / expiration ----------------------------------------------------

    async def expire_node(
        self,
        label: str,
        id: str,
        *,
        ttl_seconds: int,
    ) -> bool:
        """Mark a node for expiration via ``_expire``.

        The Memory Store ``_expire`` endpoint sets ``tombstone=true`` and
        ``valid_to=now``.  The background TTL sweeper will physically
        remove the document after the configured tombstone retention period.
        """
        body: dict[str, Any] = {
            **self._scope(label),
            "key": id,
        }
        result = await self._post("_expire", body, allow_404=True)
        if result is None:
            return False
        return result.get("expired", False)
