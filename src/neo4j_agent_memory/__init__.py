"""
Neo4j Agent Memory - A comprehensive memory system for AI agents.

This package provides a unified memory system for AI agents using Neo4j as the
persistence layer. It includes three types of memory:

- **Short-Term Memory**: Conversation history and experiences
- **Long-Term Memory**: Facts, preferences, and entities
- **Reasoning Memory**: Reasoning traces and tool usage patterns

Example usage:
    from neo4j_agent_memory import MemoryClient, MemorySettings
    from pydantic import SecretStr

    settings = MemorySettings(
        neo4j={"uri": "bolt://localhost:7687", "password": SecretStr("password")}
    )

    async with MemoryClient(settings) as client:
        # Add a message
        await client.short_term.add_message(
            session_id="user-123",
            role="user",
            content="Hi, I'm looking for Italian restaurants"
        )

        # Add a preference
        await client.long_term.add_preference(
            category="food",
            preference="I love Italian cuisine"
        )

        # Search memories
        results = await client.long_term.search_preferences("food preferences")

        # Get combined context for LLM
        context = await client.get_context("restaurant recommendation")
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from neo4j_agent_memory.config.memory_store_settings import MemoryStoreConfig
from neo4j_agent_memory.config.settings import (
    EmbeddingConfig,
    EmbeddingProvider,
    EnrichmentConfig,
    EnrichmentProvider,
    ExtractionConfig,
    ExtractorType,
    GeocodingConfig,
    GeocodingProvider,
    LLMConfig,
    LLMProvider,
    MemoryConfig,
    MemorySettings,
    Neo4jConfig,
    ResolutionConfig,
    ResolverStrategy,
    SearchConfig,
)
from neo4j_agent_memory.core.exceptions import (
    ConfigurationError,
    ConnectionError,
    EmbeddingError,
    ExtractionError,
    MemoryError,
    NotConnectedError,
    ResolutionError,
    SchemaError,
)
from neo4j_agent_memory.core.memory import BaseMemory, MemoryEntry


class GraphNode(BaseModel):
    """A node in the memory graph for visualization."""

    id: str = Field(description="Node identifier")
    labels: list[str] = Field(description="Node labels (e.g., ['Message'], ['Entity'])")
    properties: dict[str, Any] = Field(default_factory=dict, description="Node properties")


class GraphRelationship(BaseModel):
    """A relationship in the memory graph for visualization."""

    id: str = Field(description="Relationship identifier")
    type: str = Field(description="Relationship type (e.g., 'HAS_MESSAGE', 'MENTIONS')")
    from_node: str = Field(description="Source node ID")
    to_node: str = Field(description="Target node ID")
    properties: dict[str, Any] = Field(default_factory=dict, description="Relationship properties")


class MemoryGraph(BaseModel):
    """Memory graph export for visualization."""

    nodes: list[GraphNode] = Field(default_factory=list, description="Graph nodes")
    relationships: list[GraphRelationship] = Field(
        default_factory=list, description="Graph relationships"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Export metadata (filters applied, counts, etc.)",
    )


from neo4j_agent_memory.graph.backend_factory import create_backend_bundle
from neo4j_agent_memory.graph.backend_protocol import (
    BackendBundle,
    BackendCapabilities,
    GraphBackend,
    SchemaBackend,
    UnsupportedBackendOperation,
    UtilityBackend,
)
from neo4j_agent_memory.graph.client import Neo4jClient
from neo4j_agent_memory.graph.schema import SchemaManager

# Google Cloud integrations (v0.0.3+)
# These are imported conditionally to avoid requiring google dependencies.
# Stub classes provide actionable error messages when optional deps are missing.
try:
    from neo4j_agent_memory.embeddings.vertex_ai import VertexAIEmbedder
except ImportError:

    class VertexAIEmbedder:  # type: ignore[no-redef]
        """Stub for VertexAIEmbedder when google-cloud-aiplatform is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "VertexAIEmbedder requires google-cloud-aiplatform. "
                "Install with: pip install neo4j-agent-memory[vertex-ai]"
            )


try:
    from neo4j_agent_memory.integrations.google_adk import Neo4jMemoryService
except ImportError:

    class Neo4jMemoryService:  # type: ignore[no-redef]
        """Stub for Neo4jMemoryService when google-adk is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Neo4jMemoryService requires google-adk. "
                "Install with: pip install neo4j-agent-memory[google-adk]"
            )


try:
    from neo4j_agent_memory.mcp.server import Neo4jMemoryMCPServer
except ImportError:

    class Neo4jMemoryMCPServer:  # type: ignore[no-redef]
        """Stub for Neo4jMemoryMCPServer when mcp is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Neo4jMemoryMCPServer requires the mcp package. "
                "Install with: pip install neo4j-agent-memory[mcp]"
            )


from neo4j_agent_memory.memory.long_term import (
    Entity,
    EntityType,
    Fact,
    LongTermMemory,
    Preference,
    Relationship,
)
from neo4j_agent_memory.memory.reasoning import (
    ProceduralMemory,  # backward compatibility alias
    ReasoningMemory,
    ReasoningStep,
    ReasoningTrace,
    StreamingTraceRecorder,
    Tool,
    ToolCall,
    ToolCallStatus,
    ToolStats,
)
from neo4j_agent_memory.memory.short_term import (
    Conversation,
    ConversationSummary,
    Message,
    MessageRole,
    SessionInfo,
    ShortTermMemory,
)

__version__ = "0.0.5"

__all__ = [
    # Main client
    "MemoryClient",
    # Settings
    "MemorySettings",
    "Neo4jConfig",
    "MemoryStoreConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "ExtractionConfig",
    "ResolutionConfig",
    "MemoryConfig",
    "SearchConfig",
    "GeocodingConfig",
    "EnrichmentConfig",
    # Enums
    "EmbeddingProvider",
    "LLMProvider",
    "ExtractorType",
    "ResolverStrategy",
    "GeocodingProvider",
    "EnrichmentProvider",
    "MessageRole",
    "EntityType",
    "ToolCallStatus",
    # Memory types
    "ShortTermMemory",
    "LongTermMemory",
    "ReasoningMemory",
    "ProceduralMemory",  # backward compatibility alias
    # Models - Short-term
    "Message",
    "Conversation",
    "ConversationSummary",
    "SessionInfo",
    # Models - Long-term
    "Entity",
    "Preference",
    "Fact",
    "Relationship",
    # Models - Reasoning
    "ReasoningTrace",
    "ReasoningStep",
    "ToolCall",
    "ToolStats",
    "Tool",
    "StreamingTraceRecorder",
    # Base classes
    "BaseMemory",
    "MemoryEntry",
    # Graph - backend-neutral
    "GraphBackend",
    "SchemaBackend",
    "UtilityBackend",
    "BackendBundle",
    "BackendCapabilities",
    "UnsupportedBackendOperation",
    # Graph - Neo4j-specific (backend implementation detail)
    "Neo4jClient",
    "SchemaManager",
    # Graph Export
    "GraphNode",
    "GraphRelationship",
    "MemoryGraph",
    # Google Cloud integrations (v0.0.3+)
    "VertexAIEmbedder",
    "Neo4jMemoryService",
    "Neo4jMemoryMCPServer",
    # Exceptions
    "MemoryError",
    "ConnectionError",
    "SchemaError",
    "ExtractionError",
    "ResolutionError",
    "EmbeddingError",
    "ConfigurationError",
    "NotConnectedError",
]


class MemoryClient:
    """
    Main client for interacting with the Neo4j Agent Memory system.

    Provides unified access to all three memory types:
    - short_term: Conversation history and experiences
    - long_term: Facts, preferences, and entities
    - reasoning: Reasoning traces and tool usage

    Example:
        async with MemoryClient(settings) as client:
            await client.short_term.add_message(...)
            await client.long_term.add_preference(...)
            context = await client.get_context(query)
    """

    def __init__(
        self,
        settings: MemorySettings | None = None,
        *,
        embedder=None,
        extractor=None,
        resolver=None,
        geocoder=None,
        enrichment_provider=None,
    ):
        """
        Initialize the memory client.

        Args:
            settings: Memory settings (uses defaults if not provided)
            embedder: Optional embedder override (for testing)
            extractor: Optional extractor override (for testing)
            resolver: Optional resolver override (for testing)
            geocoder: Optional geocoder override (for testing)
            enrichment_provider: Optional enrichment provider override (for testing)
        """
        self._settings = settings or MemorySettings()
        self._bundle: BackendBundle | None = None
        # Legacy direct references kept for backward compat
        self._client: Neo4jClient | None = None
        self._schema_manager: SchemaManager | None = None
        self._embedder_override = embedder
        self._extractor_override = extractor
        self._resolver_override = resolver
        self._geocoder_override = geocoder
        self._enrichment_provider_override = enrichment_provider
        self._embedder = None
        self._extractor = None
        self._resolver = None
        self._geocoder = None
        self._enrichment_provider = None
        self._enrichment_service = None

        # Memory instances (initialized on connect)
        self._short_term: ShortTermMemory | None = None
        self._long_term: LongTermMemory | None = None
        self._reasoning: ReasoningMemory | None = None

    async def __aenter__(self) -> "MemoryClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """
        Connect to the configured backend and initialize memory stores.

        This sets up the database connection, creates necessary indexes
        and constraints, and initializes all memory type instances.

        The backend is selected by ``settings.backend`` (default: "neo4j").
        """
        # Create backend bundle via factory
        self._bundle = create_backend_bundle(self._settings)

        # Connect the graph backend
        await self._bundle.graph.connect()

        # Set up schema
        await self._bundle.schema.setup_all()

        # Keep legacy references for backward compatibility
        if self._bundle.backend_name == "neo4j":
            self._client = self._bundle.raw
            self._schema_manager = SchemaManager(
                self._client,
                vector_dimensions=self._settings.embedding.dimensions,
            )

        # Initialize embedder (use override if provided)
        self._embedder = self._embedder_override or self._create_embedder()

        # Initialize extractor (use override if provided)
        self._extractor = self._extractor_override or self._create_extractor()

        # Initialize resolver (use override if provided)
        self._resolver = self._resolver_override or self._create_resolver()

        # Initialize geocoder (use override if provided)
        self._geocoder = self._geocoder_override or self._create_geocoder()

        # Initialize enrichment (use override if provided)
        self._enrichment_provider = (
            self._enrichment_provider_override or self._create_enrichment_provider()
        )
        self._enrichment_service = await self._create_enrichment_service()

        # Create memory instances using the backend-neutral GraphBackend.
        self._short_term = ShortTermMemory(
            self._bundle.graph,
            self._embedder,
            self._extractor,
        )
        self._long_term = LongTermMemory(
            self._bundle.graph,
            self._embedder,
            self._extractor,
            self._resolver,
            self._geocoder,
            self._enrichment_service,
        )
        self._reasoning = ReasoningMemory(
            self._bundle.graph,
            self._embedder,
        )

    async def close(self) -> None:
        """Close the backend connection and stop background services."""
        # Stop enrichment service gracefully
        if self._enrichment_service is not None:
            await self._enrichment_service.stop()
            self._enrichment_service = None

        if self._bundle is not None:
            await self._bundle.graph.close()
            self._bundle = None
            self._client = None

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        if self._bundle is not None:
            return self._bundle.graph.is_connected
        return False

    @property
    def short_term(self) -> ShortTermMemory:
        """
        Access short-term memory (conversations, messages).

        Returns:
            ShortTermMemory instance

        Raises:
            NotConnectedError: If client is not connected
        """
        if self._short_term is None:
            raise NotConnectedError("Client not connected. Use 'async with' or call connect().")
        return self._short_term

    @property
    def long_term(self) -> LongTermMemory:
        """
        Access long-term memory (entities, preferences, facts).

        Returns:
            LongTermMemory instance

        Raises:
            NotConnectedError: If client is not connected
        """
        if self._long_term is None:
            raise NotConnectedError("Client not connected. Use 'async with' or call connect().")
        return self._long_term

    @property
    def reasoning(self) -> ReasoningMemory:
        """
        Access reasoning memory (reasoning traces, tool usage).

        Returns:
            ReasoningMemory instance

        Raises:
            NotConnectedError: If client is not connected
        """
        if self._reasoning is None:
            raise NotConnectedError("Client not connected. Use 'async with' or call connect().")
        return self._reasoning

    @property
    def schema(self) -> "SchemaBackend | SchemaManager":
        """
        Access the schema backend for database schema operations.

        For ``backend="neo4j"``, also returns a ``SchemaManager``-compatible
        object.  For other backends, returns the backend-neutral
        ``SchemaBackend``.

        Returns:
            SchemaBackend instance

        Raises:
            NotConnectedError: If client is not connected
        """
        if self._bundle is None:
            raise NotConnectedError("Client not connected. Use 'async with' or call connect().")
        return self._bundle.schema

    @property
    def graph(self) -> "Neo4jClient":
        """
        Access the underlying Neo4j graph client for custom Cypher queries.

        This allows applications to query domain-specific data stored in the
        same Neo4j database alongside agent memory operations, without creating
        a separate database connection.

        The returned client provides ``execute_read()``, ``execute_write()``,
        ``vector_search()``, and other query methods.

        .. note::

            This property is only available when ``backend="neo4j"``.
            For the ``memory_store`` backend, raw query access is not supported.
            Use ``client.neo4j`` as a clearer alternative.

        Example::

            async with MemoryClient(settings) as client:
                results = await client.graph.execute_read(
                    "MATCH (c:Customer) RETURN c.name AS name LIMIT 10"
                )

        Returns:
            Neo4jClient instance

        Raises:
            NotConnectedError: If client is not connected
            UnsupportedBackendOperation: If the backend does not support
                raw Cypher queries.
        """
        if self._bundle is None:
            raise NotConnectedError("Client not connected. Use 'async with' or call connect().")
        if self._client is None:
            raise UnsupportedBackendOperation(
                "graph (raw Cypher access)",
                self._bundle.backend_name,
                hint="Raw Cypher queries are only available with backend='neo4j'. "
                "Use the backend-neutral API methods instead.",
            )
        return self._client

    @property
    def neo4j(self) -> "Neo4jClient":
        """
        Access the underlying Neo4j client (explicit Neo4j-only property).

        Prefer this over ``client.graph`` to make the Neo4j dependency
        explicit in your code.

        Returns:
            Neo4jClient instance

        Raises:
            NotConnectedError: If client is not connected
            UnsupportedBackendOperation: If the backend is not Neo4j.
        """
        return self.graph

    @property
    def backend(self) -> BackendBundle:
        """
        Access the full backend bundle.

        Returns:
            BackendBundle with graph, schema, utility, and capabilities.

        Raises:
            NotConnectedError: If client is not connected
        """
        if self._bundle is None:
            raise NotConnectedError("Client not connected. Use 'async with' or call connect().")
        return self._bundle

    @property
    def capabilities(self) -> BackendCapabilities:
        """
        Query the current backend's capability flags.

        Returns:
            BackendCapabilities with feature flags.

        Raises:
            NotConnectedError: If client is not connected
        """
        return self.backend.capabilities

    async def get_context(
        self,
        query: str,
        *,
        session_id: str | None = None,
        include_short_term: bool = True,
        include_long_term: bool = True,
        include_reasoning: bool = True,
        max_items: int = 10,
    ) -> str:
        """
        Get combined context from all memory types for an LLM prompt.

        This method searches across all memory types and formats the results
        into a context string suitable for including in LLM prompts.

        Args:
            query: The query to search for relevant context
            session_id: Optional session ID for short-term filtering
            include_short_term: Whether to include conversation history
            include_long_term: Whether to include facts and preferences
            include_reasoning: Whether to include similar task traces
            max_items: Maximum items per memory type

        Returns:
            Formatted context string suitable for LLM prompts
        """
        parts = []

        if include_short_term:
            short_term_context = await self.short_term.get_context(
                query,
                session_id=session_id,
                max_messages=max_items,
            )
            if short_term_context:
                parts.append(f"## Conversation History\n{short_term_context}")

        if include_long_term:
            long_term_context = await self.long_term.get_context(
                query,
                max_items=max_items,
            )
            if long_term_context:
                parts.append(f"## Relevant Knowledge\n{long_term_context}")

        if include_reasoning:
            reasoning_context = await self.reasoning.get_context(
                query,
                max_traces=max_items // 2,
            )
            if reasoning_context:
                parts.append(f"## Similar Past Tasks\n{reasoning_context}")

        return "\n\n".join(parts)

    async def get_stats(self) -> dict:
        """
        Get memory statistics.

        Returns:
            Dictionary with counts for each memory type
        """
        if self._bundle is None:
            raise NotConnectedError("Client not connected.")

        return await self._bundle.utility.get_stats()

    async def get_graph(
        self,
        *,
        memory_types: list[Literal["short_term", "long_term", "reasoning"]] | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        include_embeddings: bool = False,
        limit: int = 1000,
    ) -> MemoryGraph:
        """
        Export memory graph for visualization.

        This method retrieves nodes and relationships from the memory graph,
        formatted for visualization libraries like NVL (Neo4j Visualization Library).

        Args:
            memory_types: Which memory types to include. Defaults to all.
                         Options: 'short_term', 'long_term', 'reasoning'
            session_id: Filter by session ID (for short_term and reasoning)
            since: Only include data created/updated after this time
            until: Only include data created/updated before this time
            include_embeddings: Whether to include embedding vectors in properties.
                              Set to False (default) for smaller payloads.
            limit: Maximum number of nodes to return per memory type

        Returns:
            MemoryGraph with nodes, relationships, and metadata
        """
        if self._bundle is None:
            raise NotConnectedError("Client not connected.")

        raw = await self._bundle.utility.get_graph(
            memory_types=memory_types,
            session_id=session_id,
            since=since,
            until=until,
            include_embeddings=include_embeddings,
            limit=limit,
        )

        # Convert raw dicts to Pydantic models for the public API
        nodes = [
            GraphNode(
                id=n["id"],
                labels=n.get("labels", []),
                properties=n.get("properties", {}),
            )
            for n in raw.get("nodes", [])
        ]
        relationships = [
            GraphRelationship(
                id=r["id"],
                type=r.get("type", ""),
                from_node=r.get("from_node", ""),
                to_node=r.get("to_node", ""),
                properties=r.get("properties", {}),
            )
            for r in raw.get("relationships", [])
        ]

        return MemoryGraph(
            nodes=nodes,
            relationships=relationships,
            metadata=raw.get("metadata", {}),
        )

    async def get_locations(
        self,
        *,
        session_id: str | None = None,
        has_coordinates: bool = True,
        limit: int = 500,
    ) -> list[dict]:
        """
        Get location entities, optionally filtered by conversation session.

        This method retrieves Location entities from the knowledge graph,
        with optional filtering to only include locations mentioned in a
        specific conversation (identified by session_id).

        Args:
            session_id: Filter to locations mentioned in this conversation.
                       When provided, only returns locations that have an
                       EXTRACTED_FROM relationship to messages in this session.
            has_coordinates: Only return locations with lat/lon coordinates.
                           Defaults to True for map visualization use cases.
            limit: Maximum number of locations to return. Defaults to 500.

        Returns:
            List of location dictionaries with:
                - id: Entity UUID
                - name: Location name
                - subtype: Location subtype (city, country, landmark, etc.)
                - description: Entity description
                - enriched_description: Enhanced description from enrichment
                - wikipedia_url: Wikipedia link if available
                - latitude: Latitude coordinate
                - longitude: Longitude coordinate
                - conversations: List of conversations mentioning this location
        """
        if self._bundle is None:
            raise NotConnectedError("Client not connected.")

        return await self._bundle.utility.get_locations(
            session_id=session_id,
            has_coordinates=has_coordinates,
            limit=limit,
        )

    def _create_embedder(self):
        """Create embedder based on settings."""
        config = self._settings.embedding

        if config.provider == EmbeddingProvider.OPENAI:
            from neo4j_agent_memory.embeddings.openai import OpenAIEmbedder

            return OpenAIEmbedder(
                model=config.model,
                api_key=config.api_key.get_secret_value() if config.api_key else None,
                dimensions=config.dimensions if config.dimensions != 1536 else None,
                batch_size=config.batch_size,
            )
        elif config.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            from neo4j_agent_memory.embeddings.sentence_transformers import (
                SentenceTransformerEmbedder,
            )

            return SentenceTransformerEmbedder(
                model_name=config.model,
                device=config.device,
            )
        else:
            return None

    def _create_extractor(self):
        """Create extractor based on settings.

        Uses the extraction factory to create the appropriate extractor
        based on configuration. Supports:
        - NONE: No extraction
        - LLM: LLM-based extraction (OpenAI)
        - SPACY: spaCy NER extraction (local)
        - GLINER: GLiNER zero-shot NER (local)
        - PIPELINE: Multi-stage pipeline combining multiple extractors
        """
        from neo4j_agent_memory.extraction.factory import create_extractor

        config = self._settings.extraction

        if config.extractor_type == ExtractorType.NONE:
            return None

        return create_extractor(
            extraction_config=config,
            schema_config=self._settings.schema_config,
            llm_config=self._settings.llm,
        )

    def _create_resolver(self):
        """Create resolver based on settings."""
        config = self._settings.resolution

        if config.strategy == ResolverStrategy.NONE:
            return None

        if config.strategy == ResolverStrategy.EXACT:
            from neo4j_agent_memory.resolution.exact import ExactMatchResolver

            return ExactMatchResolver()

        if config.strategy == ResolverStrategy.FUZZY:
            from neo4j_agent_memory.resolution.fuzzy import FuzzyMatchResolver

            return FuzzyMatchResolver(threshold=config.fuzzy_threshold)

        if config.strategy == ResolverStrategy.SEMANTIC:
            from neo4j_agent_memory.resolution.semantic import SemanticMatchResolver

            if self._embedder is None:
                return None
            return SemanticMatchResolver(
                self._embedder,
                threshold=config.semantic_threshold,
            )

        if config.strategy == ResolverStrategy.COMPOSITE:
            from neo4j_agent_memory.resolution.composite import CompositeResolver

            return CompositeResolver(
                embedder=self._embedder,
                exact_threshold=config.exact_threshold,
                fuzzy_threshold=config.fuzzy_threshold,
                semantic_threshold=config.semantic_threshold,
            )

        return None

    def _create_geocoder(self):
        """Create geocoder based on settings.

        Returns a configured geocoder for Location entities, or None if
        geocoding is disabled. Supports Nominatim (free, rate-limited) and
        Google (requires API key).
        """
        config = self._settings.geocoding

        if not config.enabled:
            return None

        from neo4j_agent_memory.services.geocoder import create_geocoder

        return create_geocoder(
            provider=config.provider.value,
            api_key=config.api_key.get_secret_value() if config.api_key else None,
            cache_results=config.cache_results,
            rate_limit=config.rate_limit_per_second,
            user_agent=config.user_agent,
        )

    def _create_enrichment_provider(self):
        """Create enrichment provider based on settings.

        Returns a configured enrichment provider, or None if enrichment
        is disabled. Supports Wikimedia (free) and Diffbot (requires API key).
        """
        from neo4j_agent_memory.enrichment.factory import create_enrichment_service

        return create_enrichment_service(self._settings.enrichment)

    async def _create_enrichment_service(self):
        """Create and start the background enrichment service.

        Returns a BackgroundEnrichmentService if enrichment is enabled and
        background processing is enabled, otherwise None.
        """
        if self._enrichment_provider is None:
            return None

        if not self._settings.enrichment.background_enabled:
            return None

        # Enrichment service currently requires a raw Neo4j client.
        # This will be migrated to use GraphBackend in Stage 2.
        raw_client = self._bundle.raw if self._bundle is not None else self._client
        if raw_client is None:
            return None

        from neo4j_agent_memory.enrichment.background import BackgroundEnrichmentService

        service = BackgroundEnrichmentService(
            client=raw_client,
            provider=self._enrichment_provider,
            max_queue_size=self._settings.enrichment.queue_max_size,
            max_retries=self._settings.enrichment.max_retries,
            retry_delay=self._settings.enrichment.retry_delay_seconds,
            min_confidence=self._settings.enrichment.min_confidence,
            entity_types=self._settings.enrichment.entity_types or None,
        )
        await service.start()
        return service
