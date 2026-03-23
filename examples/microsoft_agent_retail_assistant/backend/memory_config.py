"""Memory configuration for the retail assistant."""

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from neo4j_agent_memory import (
    EmbeddingConfig,
    EmbeddingProvider,
    MemoryClient,
    MemorySettings,
    MemoryStoreConfig,
)
from neo4j_agent_memory.integrations.microsoft_agent import (
    GDSAlgorithm,
    GDSConfig,
    Neo4jContextProvider,
    Neo4jMicrosoftMemory,
)

load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # Memory Store
    memory_store_endpoint: str = os.getenv("MEMORY_STORE_ENDPOINT", "https://localhost:9200")

    # AWS Bedrock
    aws_region: str = os.getenv("AWS_REGION", "us-west-2")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


def get_memory_settings() -> MemorySettings:
    """Create MemorySettings from environment."""
    return MemorySettings(
        backend="memory_store",
        memory_store=MemoryStoreConfig(
            endpoint=settings.memory_store_endpoint,
        ),
        embedding=EmbeddingConfig(
            provider=EmbeddingProvider.BEDROCK,
            model="amazon.titan-embed-text-v2:0",
            dimensions=1024,
            aws_region=settings.aws_region,
        ),
    )


def get_gds_config() -> GDSConfig:
    """Create GDS configuration for retail recommendations.

    Note: GDS (Graph Data Science) features require the Neo4j backend.
    When using backend="memory_store", GDS algorithms will fall back to
    basic alternatives if fallback_to_basic=True.
    """
    return GDSConfig(
        enabled=True,
        use_pagerank_for_ranking=True,
        pagerank_weight=0.3,
        use_community_grouping=True,
        expose_as_tools=[
            GDSAlgorithm.SHORTEST_PATH,
            GDSAlgorithm.NODE_SIMILARITY,
            GDSAlgorithm.PAGERANK,
        ],
        fallback_to_basic=True,
        warn_on_fallback=True,
    )


async def create_memory(session_id: str, user_id: str | None = None) -> Neo4jMicrosoftMemory:
    """
    Create a memory instance for a session.

    Args:
        session_id: Session identifier.
        user_id: Optional user identifier.

    Returns:
        Configured Neo4jMicrosoftMemory instance.
    """
    memory_settings = get_memory_settings()
    gds_config = get_gds_config()

    # Create memory client (in real app, manage connection lifecycle)
    client = MemoryClient(memory_settings)
    await client.connect()

    return Neo4jMicrosoftMemory(
        memory_client=client,
        session_id=session_id,
        user_id=user_id,
        include_short_term=True,
        include_long_term=True,
        include_reasoning=True,
        max_context_items=15,
        max_recent_messages=10,
        extract_entities=True,
        extract_entities_async=True,
        gds_config=gds_config,
    )


# Context provider factory for agent
def create_context_provider(
    memory_client: MemoryClient,
    session_id: str,
    user_id: str | None = None,
) -> Neo4jContextProvider:
    """
    Create a context provider for the agent.

    Args:
        memory_client: Connected MemoryClient.
        session_id: Session identifier.
        user_id: Optional user identifier.

    Returns:
        Configured Neo4jContextProvider.
    """
    return Neo4jContextProvider(
        memory_client=memory_client,
        session_id=session_id,
        user_id=user_id,
        include_short_term=True,
        include_long_term=True,
        include_reasoning=True,
        max_context_items=15,
        max_recent_messages=10,
        extract_entities=True,
        extract_entities_async=True,
        gds_config=get_gds_config(),
    )
